- extend node overlap frames 需要更新说明

reference_latents 传递方式
# 原始代码
ref_latent = vae.encode(start_image[:, :, :, :3])  # 单帧 → [1, C, 1, H, W]
positive = conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
negative = conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
关键点：
- reference_latents 是 列表，每个元素是单帧 latent [1, C, 1, H, W]
- 多张参考图 → 列表多个元素 [ref1, ref2, ...]
- 4:1 帧对应只适用于 视频序列 temporal compression
- 单帧图片编码不涉及 temporal，输出就是 1 latent frame