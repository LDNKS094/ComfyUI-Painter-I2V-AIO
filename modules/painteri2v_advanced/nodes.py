# modules/painteri2v_advanced/nodes.py
"""
PainterI2V Advanced - Full-featured Video Conditioning Node

Covers all scenarios from PainterI2V and PainterI2VExtend with high/low noise separation:
- Standard mode initial generation (I2V/FLF2V)
- Standard mode video continuation (motion latent injection)
- SVI mode initial generation (zero padding + anchor)
- SVI mode video continuation (anchor + motion from previous_latent)

High/Low noise separation:
- High noise: motion_amplitude + color_protect applied
- Low noise: original concat_latent preserved

SOURCE TRACKING: Based on PainterI2VAdvanced + Wan22FMLF SVI
"""

import torch
import comfy.model_management as mm
import comfy.utils
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    apply_motion_amplitude,
    apply_color_protect,
    apply_frequency_separation,
    apply_clip_vision,
    get_svi_padding_latent,
)


class PainterI2VAdvanced(io.ComfyNode):
    """
    Advanced Wan2.2 Video Conditioning Node with 4-cond output.

    Scenarios:
    - Initial generation: start_image → I2V, start+end → FLF2V
    - Video continuation: previous_latent → motion latent injection

    Modes:
    - Standard (svi_mode=False): Gray padding, motion amplitude enhancement
    - SVI (svi_mode=True): Zero padding (latents_mean), SVI 2.0 Pro structure
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterI2VAdvanced",
            display_name="Painter I2V Advanced",
            category="conditioning/video_models",
            inputs=[
                # Core connections
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                # Node controls
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
                    tooltip="Number of latent frames to extract from previous_latent end.",
                ),
                io.Float.Input(
                    "correct_strength",
                    default=0.01,
                    min=0.0,
                    max=0.3,
                    step=0.01,
                    tooltip="Color correction strength for color_protect.",
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
                # Optional connections
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.ClipVisionOutput.Input(
                    "clip_vision",
                    optional=True,
                    tooltip="CLIP vision output for semantic guidance.",
                ),
                io.Latent.Input(
                    "previous_latent",
                    optional=True,
                    tooltip="Previous video latent for continuation. Overrides start_image.",
                ),
                io.Latent.Input(
                    "reference_latent",
                    optional=True,
                    tooltip="External reference latent for low noise phase.",
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
        correct_strength=0.01,
        color_protect=True,
        svi_mode=False,
        start_image=None,
        end_image=None,
        clip_vision=None,
        previous_latent=None,
        reference_latent=None,
    ) -> io.NodeOutput:
        device = mm.intermediate_device()
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        H = height // spacial_scale
        W = width // spacial_scale

        # === 1. Initialize output latent ===
        latent = torch.zeros([1, latent_channels, latent_t, H, W], device=device)

        # === 2. Preprocess images and cache latents ===
        has_start = start_image is not None
        has_end = end_image is not None
        has_previous = previous_latent is not None
        anchor_start = False
        anchor_end = False
        start_latent_cached = None
        end_latent_cached = None

        if has_start:
            start_image = comfy.utils.common_upscale(
                start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            start_latent_cached = vae.encode(start_image[:, :, :, :3])
            anchor_start = True

        if has_end:
            end_image = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            end_latent_cached = vae.encode(end_image[:, :, :, :3])
            anchor_end = True

        # === 3. Handle previous_latent (overrides start_image) ===
        if has_previous:
            prev_samples = previous_latent["samples"]
            # Clamp motion_latent_count to available frames
            actual_motion_count = min(motion_latent_count, prev_samples.shape[2])
            motion_latent = prev_samples[:, :, -actual_motion_count:].clone()
            # Use first frame of previous_latent as anchor reference
            anchor_latent = prev_samples[:, :, :1].clone()
            anchor_start = True
            # Override start_latent_cached with last frame of previous
            start_latent_cached = prev_samples[:, :, -1:].clone()

        # === 4. Build concat_latent based on mode ===
        if svi_mode:
            # SVI mode: zero padding (latents_mean)
            concat_latent = get_svi_padding_latent(
                batch_size=1,
                latent_channels=latent_channels,
                latent_frames=latent_t,
                height=height,
                width=width,
                spacial_scale=spacial_scale,
                device=device,
            )

            if has_previous:
                # SVI continuation: [anchor, motion, padding]
                # Anchor at position 0
                concat_latent[:, :, :1] = anchor_latent
                # Motion latent at position 1+
                motion_end = min(1 + actual_motion_count, latent_t)
                concat_latent[:, :, 1:motion_end] = motion_latent[
                    :, :, : motion_end - 1
                ]
            elif anchor_start and start_latent_cached is not None:
                # SVI initial: anchor only
                concat_latent[:, :, :1] = start_latent_cached

            # End frame
            if anchor_end and end_latent_cached is not None:
                concat_latent[:, :, -1:] = end_latent_cached
        else:
            # Standard mode: gray padding
            image = torch.ones((length, height, width, 3), device=device) * 0.5

            if has_previous:
                # Standard continuation: motion latent injection (handled after encoding)
                pass
            elif anchor_start:
                image[0] = start_image[0, :, :, :3]

            if anchor_end:
                image[-1] = end_image[0, :, :, :3]

            concat_latent = vae.encode(image)

            # Inject motion latent for continuation
            if has_previous:
                concat_latent[:, :, :actual_motion_count] = motion_latent

        # === 5. Save original for low noise ===
        concat_latent_original = concat_latent.clone()

        # === 6. Build mask (shared by high/low noise) ===
        mask = torch.ones((1, 1, latent_t, H, W), device=device)

        # Lock first frame
        if anchor_start or has_previous:
            mask[:, :, :1] = 0.0

        # Lock end frame
        if anchor_end:
            mask[:, :, -1:] = 0.0

        # Motion latent region: soft lock (mask=1.0, default)

        # === 7. Apply motion_amplitude (high noise only) ===
        if motion_amplitude > 1.0:
            if has_start and has_end and not has_previous:
                # FLF2V mode: frequency separation
                start_l = concat_latent[:, :, 0:1]
                end_l = concat_latent[:, :, -1:]
                t = torch.linspace(0.0, 1.0, latent_t, device=device)
                t = t.view(1, 1, -1, 1, 1)
                linear_latent = start_l * (1 - t) + end_l * t

                if length > 2:
                    boost_scale = (motion_amplitude - 1.0) * 4.0
                    concat_latent = apply_frequency_separation(
                        concat_latent,
                        linear_latent,
                        boost_scale,
                        latent_channels=latent_channels,
                    )
            else:
                # I2V / continuation mode: simple amplitude boost
                concat_latent = apply_motion_amplitude(
                    concat_latent,
                    base_frame_idx=0,
                    amplitude=motion_amplitude,
                    protect_brightness=True,
                )

            # === 8. Apply color_protect (high noise only) ===
            if color_protect and correct_strength > 0:
                concat_latent = apply_color_protect(
                    concat_latent, concat_latent_original, correct_strength
                )

        # === 9. Build reference_latent ===
        auto_refs = []
        if start_latent_cached is not None:
            auto_refs.append(start_latent_cached)
        if end_latent_cached is not None:
            auto_refs.append(end_latent_cached)

        ref_latent_high = auto_refs

        # Low noise: prefer external input
        if reference_latent is not None:
            ref_latent_low = [reference_latent["samples"]]
        else:
            ref_latent_low = ref_latent_high  # Fallback to auto

        # === 10. Set conditioning ===
        # High noise (enhanced)
        positive_high = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent, "concat_mask": mask},
        )
        negative_high = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent, "concat_mask": mask},
        )

        # Low noise (original)
        positive_low = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent_original, "concat_mask": mask},
        )
        negative_low = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent_original, "concat_mask": mask},
        )

        # Add reference_latent
        if ref_latent_high:
            positive_high = node_helpers.conditioning_set_values(
                positive_high, {"reference_latents": ref_latent_high}, append=True
            )
            negative_high = node_helpers.conditioning_set_values(
                negative_high,
                {"reference_latents": [torch.zeros_like(r) for r in ref_latent_high]},
                append=True,
            )

        if ref_latent_low:
            positive_low = node_helpers.conditioning_set_values(
                positive_low, {"reference_latents": ref_latent_low}, append=True
            )
            negative_low = node_helpers.conditioning_set_values(
                negative_low,
                {"reference_latents": [torch.zeros_like(r) for r in ref_latent_low]},
                append=True,
            )

        # === 11. Add clip_vision ===
        positive_high, negative_high = apply_clip_vision(
            clip_vision, positive_high, negative_high
        )
        positive_low, negative_low = apply_clip_vision(
            clip_vision, positive_low, negative_low
        )

        out_latent = {"samples": latent}
        return io.NodeOutput(
            positive_high, negative_high, positive_low, negative_low, out_latent
        )
