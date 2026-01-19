## 对比表

| 维度 | Wan DISABLED | FLF2V |
|------|-------------|-------|
| 参考帧 | 3帧 (首/中/尾) | 2帧 (首/尾) |
| 中间帧 | ✅ 支持 | ❌ |
| Mask 系统 | 双 mask (high/low) | 单 mask |
| 强度控制 | 5个独立参数 | 无 |
| 运动增强算法 | 空间梯度调制 mask | 频率分离增强 latent |
| 增强作用位置 | mask 层面 | latent 层面 |
| 输出 | 3 conditioning | 2 conditioning |

## 核心差异

1. 运动增强方式完全不同

    | | Wan | FLF2V |
    | --- | ----- | ------- |
    | 作用对象 | mask | latent |
    | 算法 | 空间梯度 (图像差异 → mask 调制) | 频率分离 (latent 高频增强) |
    | 原理 | 运动大的区域降低 mask → 更自由生成 | 直接放大 latent 中的结构差异 |

2. 设计理念不同

    Wan: "在哪里放松控制" (mask 调制)
    FLF2V: "给什么样的参考" (latent 增强)

## 结论

不一致。两者解决同一问题（首尾帧生成）但方法论完全不同：

- Wan: 通过 mask 控制"约束程度"
- FLF2V: 通过 latent 增强"参考信息"


## 四种长视频模式对比

| 模式 | 输入依赖 | 续接机制 | 特点 |
|------|---------|----------|------|
| DISABLED | start/mid/end 图像 | 无 | 单段标准生成 |
| AUTO_CONTINUE | motion_frames (IMAGE) | 图像空间续接 | 从上段视频提取末尾帧作为首帧 |
| SVI | prev_latent + 可选图像 | Latent 空间续接 | 类似 SVI Pro，直接用 latent 条件化 |
| LATENT_CONTINUE | prev_latent | 直接 latent 注入 | 最激进，直接把上段 latent 末帧注入 |
