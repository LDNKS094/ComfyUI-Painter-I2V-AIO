import torch
import comfy.model_management as mm
import comfy.utils
import comfy.latent_formats
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    apply_motion_amplitude,
    apply_color_protect,
    apply_frequency_separation,
    extract_reference_motion,
    merge_clip_vision_outputs,
    apply_clip_vision,
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
                    "color_protect",
                    default=True,
                    tooltip="Enable color drift protection after motion amplitude enhancement.",
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
        color_protect=True,
        svi_compatible=False,
    ) -> io.NodeOutput:
        device = mm.intermediate_device()
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        H = height // spacial_scale
        W = width // spacial_scale

        # === 1. 初始化输出 latent ===
        latent = torch.zeros(
            [batch_size, latent_channels, latent_t, H, W], device=device
        )

        # === 2. 判断模式 + 预处理图像 ===
        has_start = start_image is not None
        has_end = end_image is not None
        anchor_start = False
        anchor_end = False
        start_latent_cached = None
        end_latent_cached = None

        if has_start:
            start_image = comfy.utils.common_upscale(
                start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            anchor_start = True
            start_latent_cached = vae.encode(start_image[:, :, :, :3])

        if has_end:
            end_image = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            anchor_end = True
            end_latent_cached = vae.encode(end_image[:, :, :, :3])

        # === 3. 构建 image 序列 + 编码 ===
        if has_start or has_end:
            if svi_compatible:
                # SVI 模式：用 latents_mean 填充
                concat_latent = get_svi_padding_latent(
                    batch_size=1,
                    latent_channels=latent_channels,
                    latent_frames=latent_t,
                    height=height,
                    width=width,
                    spacial_scale=spacial_scale,
                    device=device,
                )
                # 插入锚点 (使用缓存)
                if anchor_start and start_latent_cached is not None:
                    concat_latent[:, :, :1] = start_latent_cached
                if anchor_end and end_latent_cached is not None:
                    concat_latent[:, :, -1:] = end_latent_cached
            else:
                # 标准模式：灰色填充 + 编码
                image = torch.ones((length, height, width, 3), device=device) * 0.5
                if anchor_start:
                    image[0] = start_image[0, :, :, :3]
                if anchor_end:
                    image[-1] = end_image[0, :, :, :3]
                concat_latent = vae.encode(image)

            # === 4. 保存原始 concat_latent ===
            concat_latent_original = concat_latent.clone()

            # === 5. 构建 mask ===
            mask = torch.ones((1, 1, latent_t, H, W), device=device)
            if anchor_start:
                mask[:, :, :1] = 0.0
            if anchor_end:
                mask[:, :, -1:] = 0.0

            # === 6. 应用 motion_amplitude ===
            if has_start and has_end:
                # ==================== FLF2V MODE ====================
                # 计算线性插值基线
                start_l = concat_latent[:, :, 0:1]
                end_l = concat_latent[:, :, -1:]
                t = torch.linspace(0.0, 1.0, latent_t, device=device)
                t = t.view(1, 1, -1, 1, 1)
                linear_latent = start_l * (1 - t) + end_l * t

                # 频率分离 (Inverse Structural Repulsion)
                if length > 2 and motion_amplitude > 1.001:
                    boost_scale = (motion_amplitude - 1.0) * 4.0
                    concat_latent = apply_frequency_separation(
                        concat_latent,
                        linear_latent,
                        boost_scale,
                        latent_channels=latent_channels,
                    )
            else:
                # ==================== I2V MODE ====================
                # 简单差值放大
                if motion_amplitude > 1.0:
                    base_frame_idx = 0 if anchor_start else -1
                    concat_latent = apply_motion_amplitude(
                        concat_latent,
                        base_frame_idx=base_frame_idx,
                        amplitude=motion_amplitude,
                        protect_brightness=True,
                    )

            # === 7. 应用 color_protect ===
            if motion_amplitude > 1.0 and color_protect:
                concat_latent = apply_color_protect(
                    concat_latent, concat_latent_original
                )

            # === 8. 设置 conditioning ===
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent, "concat_mask": mask}
            )

            # === 9. 构建 reference_latents (使用缓存) ===
            ref_latents = []
            if start_latent_cached is not None:
                ref_latents.append(start_latent_cached)
            if end_latent_cached is not None:
                ref_latents.append(end_latent_cached)

            if ref_latents:
                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents": ref_latents}, append=True
                )
                negative = node_helpers.conditioning_set_values(
                    negative,
                    {"reference_latents": [torch.zeros_like(r) for r in ref_latents]},
                    append=True,
                )

        # === 10. 添加 reference_motion ===
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

        # === 11. 添加 clip_vision ===
        clip_vision_output = merge_clip_vision_outputs(
            clip_vision_start, clip_vision_end
        )
        positive, negative = apply_clip_vision(clip_vision_output, positive, negative)

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)
