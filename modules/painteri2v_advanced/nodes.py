# modules/painteri2v_advanced/nodes.py
"""
PainterI2V Advanced - Full-featured Video Conditioning Node

High/Low noise separation with different concat_latent content:
- High noise: start + end frame, motion_amplitude enhanced
- Low noise: start frame only, original version

Loop-compatible design:
- previous_latent accepts empty latent as fake input for ComfyUI loop
- start_image: concat source (first gen) or reference only (continuation)
- reference_latent: low noise only, from start_image encoding

Covers all scenarios from PainterI2V and PainterI2VExtend.
"""

import torch
import comfy.model_management as mm
import comfy.utils
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    apply_motion_amplitude,
    apply_color_protect,
    apply_clip_vision,
    get_svi_padding_latent,
)


class PainterI2VAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterI2VAdvanced",
            display_name="Painter I2V Advanced",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Float.Input(
                    "motion_amplitude",
                    default=1.15,
                    min=1.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Motion enhancement (high noise only).",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=4,
                    min=1,
                    max=41,
                    step=1,
                    tooltip="Pixel frames to overlap from previous video for continuation.",
                ),
                io.Float.Input(
                    "continuity_strength",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Motion frame lock strength in standard mode (0=no lock, 1=hard lock). Not used in SVI mode.",
                ),
                io.Float.Input(
                    "correct_strength",
                    default=0.01,
                    min=0.0,
                    max=0.3,
                    step=0.01,
                    tooltip="Color correction strength.",
                ),
                io.Boolean.Input(
                    "color_protect",
                    default=True,
                    tooltip="Enable color drift protection (high noise only).",
                ),
                io.Boolean.Input(
                    "svi_mode",
                    default=False,
                    tooltip="Enable SVI mode for SVI LoRA compatibility.",
                ),
                io.Image.Input("start_image", optional=True),
                io.Image.Input(
                    "end_image",
                    optional=True,
                    tooltip="End frame (high noise only).",
                ),
                io.ClipVisionOutput.Input(
                    "clip_vision",
                    optional=True,
                    tooltip="CLIP vision (low noise only).",
                ),
                io.Latent.Input(
                    "previous_latent",
                    optional=True,
                    tooltip="Previous video latent for continuation (SVI mode). Accepts empty latent for loop compatibility.",
                ),
                io.Image.Input(
                    "previous_image",
                    optional=True,
                    tooltip="Previous video frames for continuation (standard mode). Required for standard mode continuation.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="high_positive"),
                io.Conditioning.Output(display_name="high_negative"),
                io.Conditioning.Output(display_name="low_positive"),
                io.Conditioning.Output(display_name="low_negative"),
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
        motion_amplitude=1.15,
        overlap_frames=4,
        continuity_strength=0.1,
        correct_strength=0.01,
        color_protect=True,
        svi_mode=False,
        start_image=None,
        end_image=None,
        clip_vision=None,
        previous_latent=None,
        previous_image=None,
    ) -> io.NodeOutput:
        device = mm.intermediate_device()
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        H = height // spacial_scale
        W = width // spacial_scale

        latent = torch.zeros([1, latent_channels, latent_t, H, W], device=device)

        has_start = start_image is not None
        has_end = end_image is not None

        # Detect valid previous inputs
        has_previous_latent = False
        if previous_latent is not None:
            prev_samples = previous_latent["samples"]
            if prev_samples.ndim == 5 and prev_samples.shape[2] > 0:
                has_previous_latent = True

        has_previous_image = previous_image is not None and previous_image.shape[0] > 0

        # Validate: cannot have both previous_latent and previous_image
        if has_previous_latent and has_previous_image:
            raise ValueError(
                "Cannot use both previous_latent and previous_image. "
                "Use previous_latent for SVI mode, previous_image for standard mode."
            )

        # Auto-convert based on svi_mode
        if svi_mode:
            # SVI mode needs previous_latent
            if has_previous_image and not has_previous_latent:
                # Convert previous_image to latent
                prev_img_resized = comfy.utils.common_upscale(
                    previous_image.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                prev_latent_encoded = vae.encode(prev_img_resized[:, :, :, :3])
                previous_latent = {"samples": prev_latent_encoded}
                has_previous_latent = True
                has_previous_image = False
        else:
            # Standard mode needs previous_image
            if has_previous_latent and not has_previous_image:
                # Convert previous_latent to image
                previous_image = vae.decode(previous_latent["samples"])
                has_previous_image = True
                has_previous_latent = False

        # Cache for reference_latent (from start_image, used in low noise only)
        start_image_latent_for_ref = None
        start_latent_cached = None
        end_latent_cached = None
        motion_latent = None
        
        # Convert pixel frames to latent frame index (for standard mode continuity)
        overlap_latent_idx = overlap_frames // 4

        if has_start:
            start_image = comfy.utils.common_upscale(
                start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            start_latent_cached = vae.encode(start_image[:, :, :, :3])
            # Always cache for reference_latent (even in continuation mode)
            start_image_latent_for_ref = start_latent_cached

        if has_end:
            end_image = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            end_latent_cached = vae.encode(end_image[:, :, :, :3])

        # For SVI mode: extract motion_latent from previous_latent (last 1 frame only per SVI 2.0 Pro spec)
        if svi_mode and has_previous_latent:
            prev_samples = previous_latent["samples"]
            motion_latent = prev_samples[:, :, -1:].clone()
            start_latent_cached = motion_latent.clone()
            has_start = True

        if svi_mode:
            concat_high, concat_low = cls._build_svi_mode(
                start_latent=start_latent_cached,
                end_latent=end_latent_cached,
                motion_latent=motion_latent,
                has_end=has_end,
                latent_t=latent_t,
                latent_channels=latent_channels,
                H=H,
                W=W,
                spacial_scale=spacial_scale,
                height=height,
                width=width,
                device=device,
            )
        else:
            concat_high, concat_low = cls._build_standard_mode(
                vae=vae,
                start_image=start_image if not has_previous_image else None,
                end_image=end_image,
                previous_image=previous_image,
                overlap_frames=overlap_frames,
                has_start=has_start,
                has_end=has_end,
                has_previous_image=has_previous_image,
                length=length,
                height=height,
                width=width,
                device=device,
            )

        concat_high_original = concat_high.clone()

        if motion_amplitude > 1.0:
            concat_high = apply_motion_amplitude(
                concat_high,
                base_frame_idx=0,
                amplitude=motion_amplitude,
                protect_brightness=True,
            )

            if color_protect and correct_strength > 0:
                concat_high = apply_color_protect(
                    concat_high, concat_high_original, correct_strength
                )

        mask_high = torch.ones((1, 1, latent_t, H, W), device=device)
        mask_low = torch.ones((1, 1, latent_t, H, W), device=device)

        # Frame 0: hard lock
        if has_start or has_previous_image or has_previous_latent:
            mask_high[:, :, :1] = 0.0
            mask_low[:, :, :1] = 0.0

        # In standard mode with continuation: soft lock at overlap_latent_idx
        if not svi_mode and has_previous_image and overlap_latent_idx > 0 and overlap_latent_idx < latent_t:
            motion_lock = max(0.0, 1.0 - continuity_strength)
            mask_high[:, :, overlap_latent_idx:overlap_latent_idx+1] = motion_lock
            mask_low[:, :, overlap_latent_idx:overlap_latent_idx+1] = motion_lock

        # End frame lock (high noise only, fixed at 0.8 strength)
        if has_end:
            mask_high[:, :, -1:] = 0.2  # 1.0 - 0.8 = 0.2

        positive_high = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_high, "concat_mask": mask_high}
        )
        negative_high = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_high, "concat_mask": mask_high}
        )

        positive_low = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_low, "concat_mask": mask_low}
        )
        negative_low = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_low, "concat_mask": mask_low}
        )

        if start_image_latent_for_ref is not None:
            positive_low = node_helpers.conditioning_set_values(
                positive_low,
                {"reference_latents": [start_image_latent_for_ref]},
                append=True,
            )
            negative_low = node_helpers.conditioning_set_values(
                negative_low,
                {"reference_latents": [torch.zeros_like(start_image_latent_for_ref)]},
                append=True,
            )

        if clip_vision is not None:
            positive_low = node_helpers.conditioning_set_values(
                positive_low, {"clip_vision_output": clip_vision}
            )
            negative_low = node_helpers.conditioning_set_values(
                negative_low, {"clip_vision_output": clip_vision}
            )

        out_latent = {"samples": latent}
        return io.NodeOutput(
            positive_high, negative_high, positive_low, negative_low, out_latent
        )

    @classmethod
    def _build_standard_mode(
        cls,
        vae,
        start_image,
        end_image,
        previous_image,
        overlap_frames,
        has_start,
        has_end,
        has_previous_image,
        length,
        height,
        width,
        device,
    ):
        """
        Standard mode: Similar to Extend's Continuity mode.
        
        Without previous_image:
        - Frame 0: start_image
        - Frame -1: end_image (high only)
        - Other frames: grey fill
        
        With previous_image (continuation):
        - Frame 0: previous_image[-overlap_frames]
        - Frame overlap_frames: previous_image[-1]
        - Frame -1: end_image (high only)
        - Other frames: grey fill
        """
        image_high = torch.ones((length, height, width, 3), device=device) * 0.5
        image_low = torch.ones((length, height, width, 3), device=device) * 0.5

        if has_previous_image:
            # Continuation mode: use previous_image frames
            available_frames = previous_image.shape[0]
            actual_overlap = min(overlap_frames, available_frames)
            
            # Start frame: previous_image[-overlap_frames]
            start_idx = max(0, available_frames - actual_overlap)
            start_frame = previous_image[start_idx:start_idx+1].clone()
            start_frame = comfy.utils.common_upscale(
                start_frame.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image_high[0] = start_frame[0, :, :, :3]
            image_low[0] = start_frame[0, :, :, :3]
            
            # Middle frame: previous_image[-1] at position overlap_frames
            middle_idx = min(overlap_frames, length - 1)
            if middle_idx > 0:
                middle_frame = previous_image[-1:].clone()
                middle_frame = comfy.utils.common_upscale(
                    middle_frame.movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
                image_high[middle_idx] = middle_frame[0, :, :, :3]
                image_low[middle_idx] = middle_frame[0, :, :, :3]
        elif start_image is not None:
            # First generation mode: use start_image
            image_high[0] = start_image[0, :, :, :3]
            image_low[0] = start_image[0, :, :, :3]

        if end_image is not None:
            image_high[-1] = end_image[0, :, :, :3]

        concat_high = vae.encode(image_high)
        concat_low = vae.encode(image_low)

        return concat_high, concat_low

    @classmethod
    def _build_svi_mode(
        cls,
        start_latent,
        end_latent,
        motion_latent,
        has_end,
        latent_t,
        latent_channels,
        H,
        W,
        spacial_scale,
        height,
        width,
        device,
    ):
        """
        SVI 2.0 Pro mode.
        
        concat_latent = [anchor_latent, motion_latent, zero_padding]
        - anchor = start_latent (from start_image or previous)
        - motion = last 1 latent frame only (per SVI 2.0 Pro spec)
        - padding = latents_mean (zero-valued latent)
        """
        concat_high = get_svi_padding_latent(
            batch_size=1,
            latent_channels=latent_channels,
            latent_frames=latent_t,
            height=height,
            width=width,
            spacial_scale=spacial_scale,
            device=device,
        )
        concat_low = get_svi_padding_latent(
            batch_size=1,
            latent_channels=latent_channels,
            latent_frames=latent_t,
            height=height,
            width=width,
            spacial_scale=spacial_scale,
            device=device,
        )

        # Position 0: anchor_latent
        if start_latent is not None:
            concat_high[:, :, :1] = start_latent
            concat_low[:, :, :1] = start_latent

        # Position 1: motion_latent (last 1 frame only per SVI 2.0 Pro spec)
        if motion_latent is not None:
            concat_high[:, :, 1:2] = motion_latent[:, :, -1:]
            concat_low[:, :, 1:2] = motion_latent[:, :, -1:]

        # End frame (high noise only)
        if has_end and end_latent is not None:
            concat_high[:, :, -1:] = end_latent

        return concat_high, concat_low
