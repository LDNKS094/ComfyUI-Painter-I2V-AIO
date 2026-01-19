import torch
import comfy.model_management as mm
import comfy.utils
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    create_video_mask,
    apply_motion_amplitude,
    apply_frequency_separation,
    extract_reference_motion,
    merge_clip_vision_outputs,
    get_svi_padding_latent,
)


class PainterI2V(io.ComfyNode):
    """
    Unified Wan2.2 Video Conditioning Node - supports T2V, I2V, and FLF2V modes.

    Modes (auto-detected based on inputs):
    - T2V: No images → pure text-to-video conditioning
    - I2V: start_image only → image-to-video with motion amplitude fix for 4-step LoRAs
    - FLF2V: start_image + end_image → first-last-frame interpolation with inverse structural repulsion

    Key features:
    - Motion amplitude enhancement to fix slow-motion issues in accelerated models
    - Frequency separation algorithm (FLF2V mode) preserves color while boosting structure
    - Dual CLIP vision support for semantic transition guidance
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PainterI2V",
            display_name="Painter I2V",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input(
                    "motion_amplitude",
                    default=1.15,
                    min=1.0,
                    max=2.0,
                    step=0.05,
                    tooltip="1.0 = Original, 1.15 = Recommended for I2V, up to 2.0 for high-speed motion",
                ),
                io.ClipVisionOutput.Input("clip_vision_start", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.Image.Input(
                    "reference_video",
                    optional=True,
                    tooltip="Optional reference video for motion guidance. Extracts reference_motion latent.",
                ),
                io.Boolean.Input(
                    "enable_reference_latent",
                    default=True,
                    optional=True,
                    tooltip="[DEBUG] Enable reference_latents injection.",
                ),
                io.Boolean.Input(
                    "svi_compatible",
                    default=False,
                    optional=True,
                    tooltip="Enable SVI LoRA compatibility. Uses latents_mean padding instead of gray frame encoding.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        motion_amplitude,
        clip_vision_start=None,
        clip_vision_end=None,
        start_image=None,
        end_image=None,
        reference_video=None,
        enable_reference_latent=True,
        svi_compatible=False,
    ) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent_frames = ((length - 1) // 4) + 1

        # Initialize zero latent
        latent = torch.zeros(
            [
                batch_size,
                vae.latent_channels,
                latent_frames,
                height // spacial_scale,
                width // spacial_scale,
            ],
            device=mm.intermediate_device(),
        )

        # Determine mode based on inputs
        has_start = start_image is not None
        has_end = end_image is not None

        if has_start or has_end:
            # Preprocess images
            if has_start:
                start_image = comfy.utils.common_upscale(
                    start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)

            if has_end:
                end_image = comfy.utils.common_upscale(
                    end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)

            if has_start and has_end:
                # ==================== FLF2V MODE ====================
                # First-Last-Frame with Inverse Structural Repulsion
                positive, negative = cls._apply_flf2v_conditioning(
                    positive,
                    negative,
                    vae,
                    start_image,
                    end_image,
                    width,
                    height,
                    length,
                    spacial_scale,
                    latent_frames,
                    motion_amplitude,
                    enable_reference_latent,
                    svi_compatible,
                )
            else:
                # ==================== I2V MODE ====================
                # Single frame with simple difference amplification
                image = start_image if has_start else end_image
                positive, negative = cls._apply_i2v_conditioning(
                    positive,
                    negative,
                    vae,
                    image,
                    has_start,
                    width,
                    height,
                    length,
                    spacial_scale,
                    latent_frames,
                    motion_amplitude,
                    enable_reference_latent,
                    svi_compatible,
                )

        # Handle CLIP Vision outputs
        positive, negative = cls._apply_clip_vision(
            positive, negative, clip_vision_start, clip_vision_end
        )

        # Handle reference_video → reference_motion
        if reference_video is not None:
            ref_motion_latent = extract_reference_motion(
                vae, reference_video, width, height, length
            )
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_motion": ref_motion_latent}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_motion": ref_motion_latent}
            )

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)

    @classmethod
    def _apply_i2v_conditioning(
        cls,
        positive,
        negative,
        vae,
        image,
        is_start_frame,
        width,
        height,
        length,
        spacial_scale,
        latent_frames,
        motion_amplitude,
        enable_reference_latent=True,
        svi_compatible=False,
    ):
        """I2V mode: single frame anchor with motion amplitude enhancement"""

        if svi_compatible:
            # SVI mode: use latents_mean padding instead of gray frame encoding
            concat_latent_image = get_svi_padding_latent(
                batch_size=1,
                latent_channels=16,
                latent_frames=latent_frames,
                height=height,
                width=width,
                spacial_scale=spacial_scale,
                device=image.device,
            )
            # Encode anchor frame and inject at correct position
            anchor_latent = vae.encode(image[:, :, :, :3])
            if is_start_frame:
                concat_latent_image[:, :, 0:1] = anchor_latent
            else:
                concat_latent_image[:, :, -1:] = anchor_latent
        else:
            # Standard mode: gray frame encoding
            full_image = (
                torch.ones(
                    (length, height, width, image.shape[-1]),
                    device=image.device,
                    dtype=image.dtype,
                )
                * 0.5
            )

            if is_start_frame:
                full_image[0] = image[0]
            else:
                full_image[-1] = image[0]

            concat_latent_image = vae.encode(full_image[:, :, :, :3])

        # Create unified mask with sub-frame precision
        mask = create_video_mask(
            latent_frames=latent_frames,
            height=height,
            width=width,
            spacial_scale=spacial_scale,
            anchor_start=is_start_frame,
            anchor_end=not is_start_frame,
            device=image.device,
        )

        # Motion amplitude enhancement (brightness-protected)
        if motion_amplitude > 1.0:
            concat_latent_image = apply_motion_amplitude(
                concat_latent_image,
                base_frame_idx=0 if is_start_frame else -1,
                amplitude=motion_amplitude,
                protect_brightness=True,
            )

        # Inject conditioning
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        # Reference latents for style consistency (I2V mode only)
        if enable_reference_latent:
            ref_latent = vae.encode(image[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": [ref_latent]}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"reference_latents": [torch.zeros_like(ref_latent)]},
                append=True,
            )

        return positive, negative

    @classmethod
    def _apply_flf2v_conditioning(
        cls,
        positive,
        negative,
        vae,
        start_image,
        end_image,
        width,
        height,
        length,
        spacial_scale,
        latent_frames,
        motion_amplitude,
        enable_reference_latent=True,
        svi_compatible=False,
    ):
        """FLF2V mode: first-last-frame with inverse structural repulsion"""

        if svi_compatible:
            # SVI mode: use latents_mean padding for middle frames
            concat_latent_image = get_svi_padding_latent(
                batch_size=1,
                latent_channels=16,
                latent_frames=latent_frames,
                height=height,
                width=width,
                spacial_scale=spacial_scale,
            )
            # Encode and inject anchor frames
            start_latent = vae.encode(start_image[:, :, :, :3])
            end_latent = vae.encode(end_image[:, :, :, :3])
            concat_latent_image[:, :, 0:1] = start_latent
            concat_latent_image[:, :, -1:] = end_latent
            official_latent = concat_latent_image
        else:
            # Standard mode: gray frame encoding
            official_image = (
                torch.ones((length, height, width, 3), device=mm.intermediate_device())
                * 0.5
            )

            official_image[0] = start_image[0, :, :, :3]
            official_image[-1] = end_image[0, :, :, :3]

            official_latent = vae.encode(official_image)

        # Create unified mask with sub-frame precision
        mask = create_video_mask(
            latent_frames=latent_frames,
            height=height,
            width=width,
            spacial_scale=spacial_scale,
            anchor_start=True,
            anchor_end=True,
        )

        # Compute linear interpolation baseline (for detecting "slow motion" artifacts)
        start_l = official_latent[:, :, 0:1]
        end_l = official_latent[:, :, -1:]
        t = torch.linspace(
            0.0, 1.0, official_latent.shape[2], device=official_latent.device
        )
        t = t.view(1, 1, -1, 1, 1)
        linear_latent = start_l * (1 - t) + end_l * t

        # ==================== Inverse Structural Repulsion ====================
        if length > 2 and motion_amplitude > 1.001:
            # Map 1.0-2.0 input to 0.0-4.0 internal intensity
            boost_scale = (motion_amplitude - 1.0) * 4.0
            concat_latent_image = apply_frequency_separation(
                official_latent,
                linear_latent,
                boost_scale,
                latent_channels=vae.latent_channels,
            )
        else:
            concat_latent_image = official_latent

        # Inject conditioning (mask already in correct [1, 4, T, H, W] format)
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        # [DEBUG] Optional reference_latents for FLF2V - test if it helps face consistency
        if enable_reference_latent:
            ref_latent = vae.encode(start_image[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": [ref_latent]}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"reference_latents": [torch.zeros_like(ref_latent)]},
                append=True,
            )

        return positive, negative

    @classmethod
    def _apply_clip_vision(cls, positive, negative, clip_vision_start, clip_vision_end):
        """Apply CLIP vision conditioning with optional dual-clip concatenation"""

        clip_vision_output = merge_clip_vision_outputs(
            clip_vision_start, clip_vision_end
        )

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )

        return positive, negative
