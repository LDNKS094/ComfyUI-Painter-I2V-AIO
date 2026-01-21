# modules/painteri2v_extend/nodes.py
"""
PainterI2V Extend - Video Continuation Node

Dual-mode design:
- CONTINUITY mode (svi_mode=False): Start-middle frame linking for native I2V
- SVI mode (svi_mode=True): SVI 2.0 Pro architecture for SVI LoRA

SOURCE TRACKING: Based on ComfyUI-PainterLongVideo + Start-Middle Continuity discovery
"""

import torch
import comfy.model_management as mm
import comfy.utils
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    apply_motion_amplitude,
    apply_color_protect,
    get_svi_padding_latent,
)


class PainterI2VExtend(io.ComfyNode):
    """
    Video continuation node with dual-mode support.

    Modes:
    - CONTINUITY (svi_mode=False): Start-middle frame linking
      - start = previous_video[-overlap_frames]
      - middle = previous_video[-1] at position overlap_frames
      - Auto-calculated middle_strength to avoid gray artifacts

    - SVI (svi_mode=True): SVI 2.0 Pro architecture
      - anchor = anchor_image or previous_video[0]
      - motion = previous_video[-overlap_frames:] encoded
      - zero_padding with latents_mean
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
                io.Image.Input(
                    "previous_video",
                    tooltip="Previous video segment (required).",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=4,
                    min=4,
                    max=8,
                    tooltip="Overlap frames for continuity. Controls start/middle positions (CONTINUITY) or motion frames (SVI).",
                ),
                io.Float.Input(
                    "motion_amplitude",
                    default=1.15,
                    min=1.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Motion enhancement. Only applies in CONTINUITY mode.",
                ),
                io.Boolean.Input(
                    "color_protect",
                    default=True,
                    tooltip="Color drift protection. Only applies in CONTINUITY mode.",
                ),
                io.Boolean.Input(
                    "svi_mode",
                    default=False,
                    tooltip="Enable SVI mode for SVI LoRA compatibility.",
                ),
                io.Image.Input(
                    "anchor_image",
                    optional=True,
                    tooltip="Style anchor + reference_latent source. Defaults to previous_video[0].",
                ),
                io.Image.Input(
                    "end_image",
                    optional=True,
                    tooltip="Target end frame (locked).",
                ),
                io.ClipVisionOutput.Input(
                    "clip_vision",
                    optional=True,
                    tooltip="CLIP vision output for semantic guidance.",
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
        previous_video,
        overlap_frames=4,
        motion_amplitude=1.15,
        color_protect=True,
        svi_mode=False,
        anchor_image=None,
        end_image=None,
        clip_vision=None,
    ) -> io.NodeOutput:
        device = mm.intermediate_device()
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        H = height // spacial_scale
        W = width // spacial_scale

        # Initialize output latent
        latent = torch.zeros(
            [batch_size, latent_channels, latent_t, H, W], device=device
        )

        # Validate overlap_frames
        overlap_frames = min(overlap_frames, previous_video.shape[0] - 1, length - 4)
        overlap_frames = max(4, overlap_frames)

        # Preprocess end_image if provided
        has_end = end_image is not None
        end_latent_cached = None
        if has_end:
            end_image_resized = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            end_latent_cached = vae.encode(end_image_resized[:, :, :, :3])

        # Get anchor frame (for reference_latents)
        if anchor_image is not None:
            anchor_frame = comfy.utils.common_upscale(
                anchor_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        else:
            anchor_frame = comfy.utils.common_upscale(
                previous_video[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        if svi_mode:
            concat_latent, mask = cls._build_svi_mode(
                vae=vae,
                previous_video=previous_video,
                anchor_frame=anchor_frame,
                overlap_frames=overlap_frames,
                end_latent_cached=end_latent_cached,
                has_end=has_end,
                width=width,
                height=height,
                length=length,
                latent_t=latent_t,
                latent_channels=latent_channels,
                spacial_scale=spacial_scale,
                H=H,
                W=W,
                device=device,
            )
            concat_latent_original = concat_latent
        else:
            # ===== CONTINUITY MODE =====
            concat_latent, mask = cls._build_continuity_mode(
                vae=vae,
                previous_video=previous_video,
                overlap_frames=overlap_frames,
                end_latent_cached=end_latent_cached,
                has_end=has_end,
                width=width,
                height=height,
                length=length,
                latent_t=latent_t,
                H=H,
                W=W,
                device=device,
            )
            concat_latent_original = concat_latent.clone()

            # Apply motion_amplitude (CONTINUITY only)
            if motion_amplitude > 1.0:
                concat_latent = apply_motion_amplitude(
                    concat_latent,
                    base_frame_idx=0,  # Use start frame as base
                    amplitude=motion_amplitude,
                    protect_brightness=True,
                )

            # Apply color_protect (CONTINUITY only)
            if motion_amplitude > 1.0 and color_protect:
                concat_latent = apply_color_protect(
                    concat_latent, concat_latent_original
                )

        # Set conditioning
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent, "concat_mask": mask}
        )

        # Build reference_latents from anchor_frame
        ref_latent = vae.encode(anchor_frame[:, :, :, :3])
        ref_latents = [ref_latent]
        if end_latent_cached is not None:
            ref_latents.append(end_latent_cached)

        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": ref_latents}, append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"reference_latents": [torch.zeros_like(r) for r in ref_latents]},
            append=True,
        )

        # Apply clip_vision if provided
        if clip_vision is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision}
            )

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)

    @classmethod
    def _build_continuity_mode(
        cls,
        vae,
        previous_video,
        overlap_frames,
        end_latent_cached,
        has_end,
        width,
        height,
        length,
        latent_t,
        H,
        W,
        device,
    ):
        """
        CONTINUITY mode: Start-middle frame linking.

        - start = previous_video[-overlap_frames] at position 0
        - middle = previous_video[-1] at position overlap_frames
        - Auto-calculated middle_strength
        """
        middle_idx = overlap_frames

        # Extract start and middle frames
        start_image = previous_video[-overlap_frames : -overlap_frames + 1].clone()
        middle_image = previous_video[-1:].clone()

        # Resize to target dimensions
        start_image = comfy.utils.common_upscale(
            start_image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        middle_image = comfy.utils.common_upscale(
            middle_image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        # Build image tensor: gray fill with start and middle frames
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        image[0:1] = start_image[:, :, :, :3].to(device)
        image[middle_idx : middle_idx + 1] = middle_image[:, :, :, :3].to(device)

        # Encode to latent
        concat_latent = vae.encode(image)

        # Inject end_latent if provided
        if has_end and end_latent_cached is not None:
            concat_latent[:, :, -1:] = end_latent_cached

        # Build mask
        mask = torch.ones((1, 1, latent_t, H, W), device=device)

        # Lock start frame (latent frame 0)
        mask[:, :, 0:1] = 0.0

        # Lock middle frame with auto-calculated strength
        middle_latent_idx = middle_idx // 4
        middle_strength = overlap_frames * 0.025  # Auto-calculate
        middle_lock = max(0.0, 1.0 - middle_strength)
        if middle_latent_idx < latent_t:
            mask[:, :, middle_latent_idx : middle_latent_idx + 1] = middle_lock

        # Lock end frame if provided
        if has_end:
            mask[:, :, -1:] = 0.0

        return concat_latent, mask

    @classmethod
    def _build_svi_mode(
        cls,
        vae,
        previous_video,
        anchor_frame,
        overlap_frames,
        end_latent_cached,
        has_end,
        width,
        height,
        length,
        latent_t,
        latent_channels,
        spacial_scale,
        H,
        W,
        device,
        context_latents=11,
    ):
        """
        SVI mode: SVI 2.0 Pro architecture.

        concat_latent = [anchor_latent, motion_latent, zero_padding]
        - anchor = anchor_image or previous_video[0]
        - motion = last N latent frames from encoded previous_video
        - padding = latents_mean (zero-valued latent)
        """
        concat_latent = get_svi_padding_latent(
            batch_size=1,
            latent_channels=latent_channels,
            latent_frames=latent_t,
            height=height,
            width=width,
            spacial_scale=spacial_scale,
            device=device,
        )

        anchor_latent = vae.encode(anchor_frame[:, :, :, :3])

        max_pixel_frames = (context_latents - 1) * 4 + 1
        available_frames = previous_video.shape[0]
        if available_frames < max_pixel_frames:
            actual_latents = ((available_frames - 1) // 4) + 1
            max_pixel_frames = (actual_latents - 1) * 4 + 1

        prev_video_truncated = previous_video[-max_pixel_frames:]

        prev_video_resized = comfy.utils.common_upscale(
            prev_video_truncated.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        prev_latent = vae.encode(prev_video_resized[:, :, :, :3])

        motion_latent_count = ((overlap_frames - 1) // 4) + 1
        motion_latent_count = min(motion_latent_count, prev_latent.shape[2])
        motion_latent = prev_latent[:, :, -motion_latent_count:]

        concat_latent[:, :, :1] = anchor_latent

        motion_end = min(1 + motion_latent_count, latent_t)
        concat_latent[:, :, 1:motion_end] = motion_latent[:, :, : motion_end - 1]

        if has_end and end_latent_cached is not None:
            concat_latent[:, :, -1:] = end_latent_cached

        mask = torch.ones((1, 1, latent_t, H, W), device=device)
        mask[:, :, :1] = 0.0

        if has_end:
            mask[:, :, -1:] = 0.0

        return concat_latent, mask
