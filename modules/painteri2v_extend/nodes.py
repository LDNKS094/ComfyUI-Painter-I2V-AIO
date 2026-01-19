# modules/painteri2v_extend/nodes.py
"""
PainterI2V Extend - Video Continuation Node

Specialized for extending/continuing videos from a previous segment.
Uses motion_frames overlap for smooth transitions.

SOURCE TRACKING: Based on ComfyUI-PainterLongVideo, refactored for new API
"""

import torch
import comfy.model_management as mm
import comfy.utils
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    create_video_mask,
    apply_motion_amplitude,
    extract_reference_motion,
    merge_clip_vision_outputs,
    get_svi_padding_latent,
)


class PainterI2VExtend(io.ComfyNode):
    """
    Video continuation node for extending previous video segments.

    Key features:
    - motion_frames overlap for smooth transitions
    - Automatic reference_latents from previous_video last frame
    - Optional end_image for FLF-style continuation
    - Optional reference_video for motion guidance (explicit, not from previous_video)
    - SVI LoRA compatibility mode
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterI2VExtend",
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
                    tooltip="1.0 = Original, 1.15 = Recommended, up to 2.0 for high-speed motion",
                ),
                io.Image.Input(
                    "previous_video",
                    tooltip="Previous video segment (required). Last frame used as anchor.",
                ),
                io.Int.Input(
                    "motion_frames",
                    default=5,
                    min=1,
                    max=20,
                    tooltip="Number of overlap frames from previous_video for motion continuity.",
                ),
                io.Image.Input(
                    "end_image",
                    optional=True,
                    tooltip="Optional target end frame for FLF-style continuation.",
                ),
                io.Image.Input(
                    "reference_video",
                    optional=True,
                    tooltip="Optional reference video for motion guidance. NOT extracted from previous_video.",
                ),
                io.ClipVisionOutput.Input("clip_vision", optional=True),
                io.Boolean.Input(
                    "enable_reference_latent",
                    default=True,
                    optional=True,
                    tooltip="Enable reference_latents injection from previous_video last frame.",
                ),
                io.Boolean.Input(
                    "svi_compatible",
                    default=False,
                    optional=True,
                    tooltip="Enable SVI LoRA compatibility. Uses latents_mean padding.",
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
        previous_video,
        motion_frames=5,
        end_image=None,
        reference_video=None,
        clip_vision=None,
        enable_reference_latent=True,
        svi_compatible=False,
    ) -> io.NodeOutput:
        device = mm.intermediate_device()
        spacial_scale = vae.spacial_compression_encode()
        latent_frames = ((length - 1) // 4) + 1

        # Initialize output latent
        latent = torch.zeros(
            [
                batch_size,
                vae.latent_channels,
                latent_frames,
                height // spacial_scale,
                width // spacial_scale,
            ],
            device=device,
        )

        # Extract and resize last frame from previous_video as anchor
        last_frame = previous_video[-1:].clone()
        last_frame = comfy.utils.common_upscale(
            last_frame.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        has_end = end_image is not None

        if has_end:
            # FLF-style continuation: start from previous + target end
            end_image = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

            positive, negative = cls._apply_flf_extend_conditioning(
                positive,
                negative,
                vae,
                last_frame,
                end_image,
                width,
                height,
                length,
                spacial_scale,
                latent_frames,
                motion_amplitude,
                svi_compatible,
            )
        else:
            # Standard continuation: only start anchor
            positive, negative = cls._apply_extend_conditioning(
                positive,
                negative,
                vae,
                last_frame,
                width,
                height,
                length,
                spacial_scale,
                latent_frames,
                motion_amplitude,
                svi_compatible,
            )

        # Reference latents from previous_video last frame
        if enable_reference_latent:
            ref_latent = vae.encode(last_frame[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": [ref_latent]}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"reference_latents": [torch.zeros_like(ref_latent)]},
                append=True,
            )

        # Reference motion from explicit reference_video (NOT from previous_video)
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

        # CLIP vision
        if clip_vision is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"clip_vision_output": clip_vision}
            )

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)

    @classmethod
    def _apply_extend_conditioning(
        cls,
        positive,
        negative,
        vae,
        start_frame,
        width,
        height,
        length,
        spacial_scale,
        latent_frames,
        motion_amplitude,
        svi_compatible=False,
    ):
        """Standard extend mode: single start anchor"""

        if svi_compatible:
            # SVI mode: latents_mean padding
            concat_latent_image = get_svi_padding_latent(
                batch_size=1,
                latent_channels=16,
                latent_frames=latent_frames,
                height=height,
                width=width,
                spacial_scale=spacial_scale,
                device=start_frame.device,
            )
            anchor_latent = vae.encode(start_frame[:, :, :, :3])
            concat_latent_image[:, :, 0:1] = anchor_latent
        else:
            # Standard mode: gray frame encoding
            image_seq = (
                torch.ones(
                    (length, height, width, start_frame.shape[-1]),
                    device=start_frame.device,
                    dtype=start_frame.dtype,
                )
                * 0.5
            )
            image_seq[0] = start_frame[0]
            concat_latent_image = vae.encode(image_seq[:, :, :, :3])

        # Create unified mask with sub-frame precision
        mask = create_video_mask(
            latent_frames=latent_frames,
            height=height,
            width=width,
            spacial_scale=spacial_scale,
            anchor_start=True,
            anchor_end=False,
            device=start_frame.device,
        )

        # Motion amplitude enhancement
        if motion_amplitude > 1.0:
            concat_latent_image = apply_motion_amplitude(
                concat_latent_image,
                base_frame_idx=0,
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

        return positive, negative

    @classmethod
    def _apply_flf_extend_conditioning(
        cls,
        positive,
        negative,
        vae,
        start_frame,
        end_frame,
        width,
        height,
        length,
        spacial_scale,
        latent_frames,
        motion_amplitude,
        svi_compatible=False,
    ):
        """FLF-style extend mode: start + end anchors"""

        if svi_compatible:
            # SVI mode: latents_mean padding
            concat_latent_image = get_svi_padding_latent(
                batch_size=1,
                latent_channels=16,
                latent_frames=latent_frames,
                height=height,
                width=width,
                spacial_scale=spacial_scale,
            )
            start_latent = vae.encode(start_frame[:, :, :, :3])
            end_latent = vae.encode(end_frame[:, :, :, :3])
            concat_latent_image[:, :, 0:1] = start_latent
            concat_latent_image[:, :, -1:] = end_latent
        else:
            # Standard mode: gray frame encoding
            image_seq = (
                torch.ones(
                    (length, height, width, 3),
                    device=mm.intermediate_device(),
                    dtype=torch.float32,
                )
                * 0.5
            )
            image_seq[0] = start_frame[0, :, :, :3]
            image_seq[-1] = end_frame[0, :, :, :3]
            concat_latent_image = vae.encode(image_seq)

        # Create unified mask with sub-frame precision
        mask = create_video_mask(
            latent_frames=latent_frames,
            height=height,
            width=width,
            spacial_scale=spacial_scale,
            anchor_start=True,
            anchor_end=True,
        )

        # Inject conditioning
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        return positive, negative


# Node registration
NODE_CLASS_MAPPINGS = {
    "PainterI2VExtend": PainterI2VExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2VExtend": "PainterI2V Extend (Video Continuation)",
}
