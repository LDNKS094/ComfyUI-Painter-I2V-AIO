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
    apply_motion_amplitude,
    extract_reference_motion,
    merge_clip_vision_outputs,
    apply_clip_vision,
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
            display_name="Painter I2V Extend",
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
                io.ClipVisionOutput.Input("clip_vision_start", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end", optional=True),
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
                io.Boolean.Input(
                    "enable_reference_latent",
                    default=True,
                    optional=True,
                    tooltip="[DEBUG] Enable reference_latents injection from previous_video last frame.",
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
        clip_vision_start=None,
        clip_vision_end=None,
        motion_frames=5,
        end_image=None,
        reference_video=None,
        enable_reference_latent=True,
        svi_compatible=False,
    ) -> io.NodeOutput:
        device = mm.intermediate_device()
        spacial_scale = vae.spacial_compression_encode()
        latent_frames = ((length - 1) // 4) + 1

        # Validate motion_frames
        actual_motion_frames = min(motion_frames, previous_video.shape[0], length - 1)
        if actual_motion_frames < 1:
            actual_motion_frames = 1

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

        # Extract overlap frames from previous_video (AUTO_CONTINUE mechanism)
        overlap_frames = previous_video[-actual_motion_frames:].clone()
        overlap_frames = comfy.utils.common_upscale(
            overlap_frames.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        has_end = end_image is not None

        if has_end:
            # FLF-style continuation: overlap frames + target end
            end_image = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

            positive, negative = cls._apply_flf_extend_conditioning(
                positive,
                negative,
                vae,
                overlap_frames,
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
            # Standard continuation: overlap frames only
            positive, negative = cls._apply_extend_conditioning(
                positive,
                negative,
                vae,
                overlap_frames,
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
            # Use the last frame of overlap as reference
            last_frame = overlap_frames[-1:]
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

        # CLIP vision - merge start and end if both provided
        clip_vision_output = merge_clip_vision_outputs(
            clip_vision_start, clip_vision_end
        )
        positive, negative = apply_clip_vision(clip_vision_output, positive, negative)

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)

    @classmethod
    def _apply_extend_conditioning(
        cls,
        positive,
        negative,
        vae,
        overlap_frames,
        width,
        height,
        length,
        spacial_scale,
        latent_frames,
        motion_amplitude,
        svi_compatible=False,
    ):
        """
        Standard extend mode with AUTO_CONTINUE mechanism.

        Fills overlap_frames at the beginning of the sequence and hard-locks them (mask=0).
        This ensures motion continuity at the junction point.
        """
        num_overlap = overlap_frames.shape[0]
        motion_latent_frames = ((num_overlap - 1) // 4) + 1

        if svi_compatible:
            # SVI mode: latents_mean padding
            concat_latent_image = get_svi_padding_latent(
                batch_size=1,
                latent_channels=16,
                latent_frames=latent_frames,
                height=height,
                width=width,
                spacial_scale=spacial_scale,
                device=overlap_frames.device,
            )
            # Encode overlap frames and insert at beginning
            overlap_latent = vae.encode(overlap_frames[:, :, :, :3])
            concat_latent_image[:, :, :motion_latent_frames] = overlap_latent[
                :, :, :motion_latent_frames
            ]
        else:
            # Standard mode: build image sequence with overlap frames at beginning
            image_seq = (
                torch.ones(
                    (length, height, width, overlap_frames.shape[-1]),
                    device=overlap_frames.device,
                    dtype=overlap_frames.dtype,
                )
                * 0.5
            )
            # Fill overlap frames at the beginning
            image_seq[:num_overlap] = overlap_frames
            concat_latent_image = vae.encode(image_seq[:, :, :, :3])

        # Create mask: hard-lock the overlap region (mask=0)
        mask = torch.ones(
            (1, 4, latent_frames, height // spacial_scale, width // spacial_scale),
            device=overlap_frames.device,
            dtype=overlap_frames.dtype,
        )
        # Lock the motion_frames region
        mask[:, :, :motion_latent_frames] = 0.0

        # Motion amplitude enhancement (apply only to non-locked region)
        if motion_amplitude > 1.0 and not svi_compatible:
            concat_latent_image = apply_motion_amplitude(
                concat_latent_image,
                base_frame_idx=num_overlap - 1,  # Use last overlap frame as base
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
        overlap_frames,
        end_frame,
        width,
        height,
        length,
        spacial_scale,
        latent_frames,
        motion_amplitude,
        svi_compatible=False,
    ):
        """
        FLF-style extend mode with AUTO_CONTINUE mechanism.

        Fills overlap_frames at the beginning + end_frame at the end.
        Hard-locks both regions (mask=0).
        """
        num_overlap = overlap_frames.shape[0]
        motion_latent_frames = ((num_overlap - 1) // 4) + 1

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
            # Encode and insert overlap frames at beginning
            overlap_latent = vae.encode(overlap_frames[:, :, :, :3])
            concat_latent_image[:, :, :motion_latent_frames] = overlap_latent[
                :, :, :motion_latent_frames
            ]
            # Encode and insert end frame
            end_latent = vae.encode(end_frame[:, :, :, :3])
            concat_latent_image[:, :, -1:] = end_latent
        else:
            # Standard mode: build image sequence
            image_seq = (
                torch.ones(
                    (length, height, width, 3),
                    device=mm.intermediate_device(),
                    dtype=torch.float32,
                )
                * 0.5
            )
            # Fill overlap frames at the beginning
            image_seq[:num_overlap] = overlap_frames[:, :, :, :3]
            # Fill end frame
            image_seq[-1] = end_frame[0, :, :, :3]
            concat_latent_image = vae.encode(image_seq)

        # Create mask: hard-lock overlap region + end frame
        mask = torch.ones(
            (1, 4, latent_frames, height // spacial_scale, width // spacial_scale),
            device=concat_latent_image.device,
            dtype=concat_latent_image.dtype,
        )
        # Lock the motion_frames region at start
        mask[:, :, :motion_latent_frames] = 0.0
        # Lock the end frame
        mask[:, :, -1:] = 0.0

        # Inject conditioning
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        return positive, negative
