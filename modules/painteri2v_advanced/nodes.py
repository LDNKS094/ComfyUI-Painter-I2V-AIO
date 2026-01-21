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
                    "motion_latent_count",
                    default=1,
                    min=1,
                    max=11,
                    step=1,
                    tooltip="Latent frames to extract from previous_latent end.",
                ),
                io.Float.Input(
                    "high_noise_end_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="End frame lock strength for high noise (1.0=hard lock).",
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
                    tooltip="Previous video latent for continuation. Accepts empty latent for loop compatibility.",
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
        motion_latent_count=1,
        high_noise_end_strength=1.0,
        correct_strength=0.01,
        color_protect=True,
        svi_mode=False,
        start_image=None,
        end_image=None,
        clip_vision=None,
        previous_latent=None,
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

        # Detect fake previous_latent (empty latent for loop compatibility)
        has_previous = False
        if previous_latent is not None:
            prev_samples = previous_latent["samples"]
            # Check if it's a real latent (has temporal frames)
            if prev_samples.numel() > 0 and prev_samples.shape[2] > 0:
                has_previous = True

        # Cache for reference_latent (from start_image, used in low noise only)
        start_image_latent_for_ref = None
        start_latent_cached = None
        end_latent_cached = None
        motion_latent = None
        actual_motion_count = 0

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

        if has_previous:
            prev_samples = previous_latent["samples"]
            actual_motion_count = min(motion_latent_count, prev_samples.shape[2])
            motion_latent = prev_samples[:, :, -actual_motion_count:].clone()
            start_latent_cached = prev_samples[:, :, -1:].clone()
            has_start = True

        if svi_mode:
            concat_high, concat_low = cls._build_svi_mode(
                start_latent=start_latent_cached,
                end_latent=end_latent_cached,
                motion_latent=motion_latent,
                actual_motion_count=actual_motion_count,
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
                start_image=start_image if not has_previous else None,
                end_image=end_image,
                start_latent=start_latent_cached,
                end_latent=end_latent_cached,
                motion_latent=motion_latent,
                actual_motion_count=actual_motion_count,
                has_start=has_start,
                has_end=has_end,
                has_previous=has_previous,
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

        if has_start or has_previous:
            mask_high[:, :, :1] = 0.0
            mask_low[:, :, :1] = 0.0

        if has_end:
            mask_high[:, :, -1:] = max(0.0, 1.0 - high_noise_end_strength)

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
        start_latent,
        end_latent,
        motion_latent,
        actual_motion_count,
        has_start,
        has_end,
        has_previous,
        length,
        height,
        width,
        device,
    ):
        image_high = torch.ones((length, height, width, 3), device=device) * 0.5
        image_low = torch.ones((length, height, width, 3), device=device) * 0.5

        if start_image is not None and not has_previous:
            image_high[0] = start_image[0, :, :, :3]
            image_low[0] = start_image[0, :, :, :3]

        if end_image is not None:
            image_high[-1] = end_image[0, :, :, :3]

        concat_high = vae.encode(image_high)
        concat_low = vae.encode(image_low)

        if motion_latent is not None:
            concat_high[:, :, :actual_motion_count] = motion_latent
            concat_low[:, :, :actual_motion_count] = motion_latent

        return concat_high, concat_low

    @classmethod
    def _build_svi_mode(
        cls,
        start_latent,
        end_latent,
        motion_latent,
        actual_motion_count,
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

        if start_latent is not None:
            concat_high[:, :, :1] = start_latent
            concat_low[:, :, :1] = start_latent

        if has_end and end_latent is not None:
            concat_high[:, :, -1:] = end_latent

        if motion_latent is not None:
            concat_high[:, :, :actual_motion_count] = motion_latent
            concat_low[:, :, :actual_motion_count] = motion_latent

        return concat_high, concat_low
