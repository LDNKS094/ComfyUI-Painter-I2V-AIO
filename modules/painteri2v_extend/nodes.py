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
import comfy.latent_formats
import node_helpers
from comfy_api.latest import io

from ..common.utils import (
    apply_motion_amplitude,
    apply_color_protect,
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
                    "color_protect",
                    default=True,
                    tooltip="Enable color drift protection after motion amplitude enhancement.",
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

        # === 2. 提取 overlap_frames + upscale ===
        actual_motion_frames = min(motion_frames, previous_video.shape[0], length - 1)
        if actual_motion_frames < 1:
            actual_motion_frames = 1

        overlap_frames = previous_video[-actual_motion_frames:].clone()
        overlap_frames = comfy.utils.common_upscale(
            overlap_frames.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        num_overlap = overlap_frames.shape[0]
        motion_latent_frames = ((num_overlap - 1) // 4) + 1

        # === 3. 预处理 end_image (如有) + 缓存编码 ===
        has_end = end_image is not None
        end_latent_cached = None

        if has_end:
            end_image = comfy.utils.common_upscale(
                end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            end_latent_cached = vae.encode(end_image[:, :, :, :3])

        # === 4. 构建 image 序列 + 编码 ===
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
            # 编码 overlap 并插入开头
            overlap_latent = vae.encode(overlap_frames[:, :, :, :3])
            concat_latent[:, :, :motion_latent_frames] = overlap_latent[
                :, :, :motion_latent_frames
            ]
            # 插入 end_image (如有)
            if has_end and end_latent_cached is not None:
                concat_latent[:, :, -1:] = end_latent_cached
        else:
            # 标准模式：灰色填充
            image = torch.ones((length, height, width, 3), device=device) * 0.5
            # 填充 overlap frames 到开头
            image[:num_overlap] = overlap_frames[:, :, :, :3]
            # 填充 end_image (如有)
            if has_end:
                image[-1] = end_image[0, :, :, :3]
            concat_latent = vae.encode(image)

        # === 5. 保存原始 concat_latent ===
        concat_latent_original = concat_latent.clone()

        # === 6. 构建 mask ===
        # 硬锁定首帧，motion_frames 后续帧作为软参考（mask=1.0，由 concat_latent 提供运动信息）
        mask = torch.ones((1, 1, latent_t, H, W), device=device)
        mask[:, :, :1] = 0.0  # 只硬锁首帧
        # 锁定 end_frame (如有)
        if has_end:
            mask[:, :, -1:] = 0.0

        # === 7. 应用 motion_amplitude ===
        if motion_amplitude > 1.0 and not svi_compatible:
            concat_latent = apply_motion_amplitude(
                concat_latent,
                base_frame_idx=num_overlap - 1,  # 用最后一个 overlap frame 作为基准
                amplitude=motion_amplitude,
                protect_brightness=True,
            )

        # === 8. 应用 color_protect ===
        if motion_amplitude > 1.0 and color_protect and not svi_compatible:
            concat_latent = apply_color_protect(concat_latent, concat_latent_original)

        # === 9. 设置 conditioning ===
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent, "concat_mask": mask}
        )

        # === 10. 构建 reference_latents ===
        # 使用 overlap 最后一帧作为 reference
        last_frame = overlap_frames[-1:]
        ref_latent = vae.encode(last_frame[:, :, :, :3])

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

        # === 11. 添加 reference_motion ===
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

        # === 12. 添加 clip_vision ===
        clip_vision_output = merge_clip_vision_outputs(
            clip_vision_start, clip_vision_end
        )
        positive, negative = apply_clip_vision(clip_vision_output, positive, negative)

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, out_latent)
