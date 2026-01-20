import torch
import comfy.model_management as mm
import comfy.utils
import comfy.latent_formats
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    apply_motion_amplitude,
    apply_color_protect,
    merge_clip_vision_outputs,
)


class PainterI2VAdvanced(io.ComfyNode):
    """Advanced Wan2.2 I2V node with prev_latent continuation, motion amplitude enhancement, and color protection."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterI2VAdvanced",
            display_name="Painter I2V Advanced",
            category="conditioning/video_models",
            inputs=[
                # 核心连接（必须）
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                # 节点控件
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Float.Input(
                    "motion_amplitude", default=1.15, min=1.0, max=2.0, step=0.05
                ),
                io.Int.Input("motion_frames", default=5, min=1, max=20, step=1),
                io.Float.Input(
                    "correct_strength", default=0.01, min=0.0, max=0.3, step=0.01
                ),
                io.Boolean.Input("color_protect", default=True),
                io.Boolean.Input("svi_compatible", default=False),
                # 可选连接
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_start", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end", optional=True),
                io.Latent.Input("prev_latent", optional=True),
                io.Latent.Input("reference_latents", optional=True),
                io.Latent.Input("reference_motion", optional=True),
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
        motion_frames=5,
        correct_strength=0.01,
        color_protect=True,
        svi_compatible=False,
        start_image=None,
        end_image=None,
        clip_vision_start=None,
        clip_vision_end=None,
        prev_latent=None,
        reference_latents=None,
        reference_motion=None,
    ) -> io.NodeOutput:
        device = mm.intermediate_device()
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        H = height // spacial_scale
        W = width // spacial_scale

        # === 1. 初始化输出 latent ===
        latent = torch.zeros([1, latent_channels, latent_t, H, W], device=device)

        # === 2. 构建 image 序列 (灰色填充) ===
        image = torch.ones((length, height, width, 3), device=device) * 0.5

        # === 3. 处理锚点图像 + 缓存编码结果 ===
        anchor_start = False
        anchor_end = False
        motion_latent_count = 0
        start_latent_cached = None
        end_latent_cached = None

        # prev_latent 覆盖 start_image
        if prev_latent is not None:
            motion_latent_count = min(
                prev_latent["samples"].shape[2], ((motion_frames - 1) // 4) + 1
            )
            anchor_start = True
            # start_image 被忽略，首帧 latent 来自 prev_latent
            start_latent_cached = prev_latent["samples"][:, :, -1:]
        elif start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image[0] = start_image[0, :, :, :3]
            anchor_start = True
            # 缓存 start_image 编码
            start_latent_cached = vae.encode(start_image[:, :, :, :3])

        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image[-1] = end_image[0, :, :, :3]
            anchor_end = True
            # 缓存 end_image 编码
            end_latent_cached = vae.encode(end_image[:, :, :, :3])

        # === 4. 编码基础序列 ===
        if svi_compatible:
            # SVI 模式：用 latents_mean 填充
            concat_latent = torch.zeros(
                1, latent_channels, latent_t, H, W, device=device, dtype=torch.float32
            )
            concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)

            # 插入锚点 (使用缓存)
            if anchor_start and prev_latent is None and start_latent_cached is not None:
                concat_latent[:, :, :1] = start_latent_cached
            if anchor_end and end_latent_cached is not None:
                concat_latent[:, :, -1:] = end_latent_cached
        else:
            # 标准模式：编码整个 image 序列
            concat_latent = vae.encode(image)

        # === 5. 注入 motion_latent (prev_latent 覆盖模式) ===
        if prev_latent is not None:
            motion_latent = prev_latent["samples"][:, :, -motion_latent_count:].clone()
            concat_latent[:, :, :motion_latent_count] = motion_latent

        # 保存原始版本用于低噪
        concat_latent_original = concat_latent.clone()

        # === 6. 构建 mask (高/低噪共用) ===
        mask = torch.ones((1, 1, latent_t, H, W), device=device)

        # 首帧硬锁
        if anchor_start:
            mask[:, :, :1] = 0.0

        # 尾帧硬锁
        if anchor_end:
            mask[:, :, -1:] = 0.0

        # motion_latent 区域软锁 (mask=1.0, 已经是默认值)

        # === 7. 应用 motion_amplitude (仅高噪) ===
        if motion_amplitude > 1.0:
            concat_latent = apply_motion_amplitude(
                concat_latent, base_frame_idx=0, amplitude=motion_amplitude
            )

            # === 8. 应用 color_protect (仅高噪) ===
            if color_protect and correct_strength > 0:
                concat_latent = apply_color_protect(
                    concat_latent, concat_latent_original, correct_strength
                )

        # === 9. 构建 reference_latents (使用缓存) ===
        auto_refs = []
        if start_latent_cached is not None:
            auto_refs.append(start_latent_cached)
        if end_latent_cached is not None:
            auto_refs.append(end_latent_cached)

        ref_latents_high = auto_refs

        # 低噪：优先外部输入
        if reference_latents is not None:
            ref_latents_low = [reference_latents["samples"]]
        else:
            ref_latents_low = ref_latents_high  # 复用高噪

        # === 10. 设置 conditioning ===
        # 高噪 (增强版)
        positive_high = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent, "concat_mask": mask},
        )
        negative_high = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent, "concat_mask": mask},
        )

        # 低噪 (原始版)
        positive_low = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent_original, "concat_mask": mask},
        )
        negative_low = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent_original, "concat_mask": mask},
        )

        # 添加 reference_latents
        if ref_latents_high:
            positive_high = node_helpers.conditioning_set_values(
                positive_high, {"reference_latents": ref_latents_high}, append=True
            )
            negative_high = node_helpers.conditioning_set_values(
                negative_high,
                {"reference_latents": [torch.zeros_like(r) for r in ref_latents_high]},
                append=True,
            )

        if ref_latents_low:
            positive_low = node_helpers.conditioning_set_values(
                positive_low, {"reference_latents": ref_latents_low}, append=True
            )
            negative_low = node_helpers.conditioning_set_values(
                negative_low,
                {"reference_latents": [torch.zeros_like(r) for r in ref_latents_low]},
                append=True,
            )

        # === 11. 添加 reference_motion (有输入即启用) ===
        if reference_motion is not None:
            ref_motion = reference_motion["samples"]
            positive_high = node_helpers.conditioning_set_values(
                positive_high, {"reference_motion": ref_motion}
            )
            positive_low = node_helpers.conditioning_set_values(
                positive_low, {"reference_motion": ref_motion}
            )

        # === 12. 添加 clip_vision ===
        clip_vision_output = merge_clip_vision_outputs(
            clip_vision_start, clip_vision_end
        )
        if clip_vision_output is not None:
            positive_high = node_helpers.conditioning_set_values(
                positive_high, {"clip_vision_output": clip_vision_output}
            )
            negative_high = node_helpers.conditioning_set_values(
                negative_high, {"clip_vision_output": clip_vision_output}
            )
            positive_low = node_helpers.conditioning_set_values(
                positive_low, {"clip_vision_output": clip_vision_output}
            )
            negative_low = node_helpers.conditioning_set_values(
                negative_low, {"clip_vision_output": clip_vision_output}
            )

        out_latent = {"samples": latent}
        return io.NodeOutput(
            positive_high, negative_high, positive_low, negative_low, out_latent
        )
