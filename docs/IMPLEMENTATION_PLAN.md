# PainterI2V 节点系列实现规划

## 概述

三个节点，分层设计：

| 节点 | 定位 | 复杂度 | 输出 |
|------|------|--------|------|
| PainterI2V | 首发生成 | 低 | 2 cond |
| PainterI2V Extend | 视频续接 | 低 | 2 cond |
| PainterI2V Advanced | 全功能 | 高 | 4 cond |

---

## 节点 1: PainterI2V

### 定位
单段视频首发生成，入门级

### 输入

#### 核心连接（必须）

| 参数 | 类型 | 说明 |
|------|------|------|
| positive / negative | CONDITIONING | |
| vae | VAE | |

#### 节点控件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| width / height | INT | 832 / 480 | |
| length | INT | 81 | 生成帧数 |
| motion_amplitude | FLOAT | 1.15 | 动作幅度增强 |
| svi_compatible | BOOLEAN | False | 兼容 SVI 采样器 |

#### 可选连接

| 参数 | 类型 | 说明 |
|------|------|------|
| start_image | IMAGE | 首帧 |
| end_image | IMAGE | 尾帧 |
| reference_video | IMAGE | 动作参考 |
| clip_vision_start / end | CLIP_VISION_OUTPUT | |

### 内部行为
- reference_latent 总是启用（从 start/end image 自动生成）

### 输出
`positive`, `negative`, `latent` (2 cond)

### 模式自动切换

| 输入 | 模式 |
|------|------|
| 无图像 | T2V |
| start_image | I2V |
| start + end | FLF2V |

### 来源
原 PainterI2V + PainterFLF2V 合并

---

## 节点 2: PainterI2V Extend

### 定位
视频续接专用，入门级

### 输入

#### 核心连接（必须）

| 参数 | 类型 | 说明 |
|------|------|------|
| positive / negative | CONDITIONING | |
| vae | VAE | |
| previous_video | IMAGE | 前置视频 |

#### 节点控件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| width / height | INT | 832 / 480 | |
| length | INT | 81 | 生成帧数 |
| motion_amplitude | FLOAT | 1.15 | 动作幅度增强 |
| motion_frames | INT | 5 | 重叠帧数 |
| svi_compatible | BOOLEAN | False | 兼容 SVI 采样器 |

#### 可选连接

| 参数 | 类型 | 说明 |
|------|------|------|
| end_image | IMAGE | 目标尾帧 |
| reference_video | IMAGE | 动作参考 |
| clip_vision_start / end | CLIP_VISION_OUTPUT | |

### 内部行为
- reference_latent 总是启用（从 previous_video 末帧 / end_image 自动生成）

### 输出
`positive`, `negative`, `latent` (2 cond)

### 核心机制：AUTO_CONTINUE

```
previous_video[-motion_frames:] → 填入序列开头 → mask=0 硬锁定
```

- 高/低噪使用相同 conditioning
- 后处理直接裁剪重叠帧（无需混合）
- reference_motion 仅从 reference_video 提取，不从 previous_video 隐式提取

### 来源
原 PainterLongVideo + Wan AUTO_CONTINUE

---

## 节点 3: PainterI2V Advanced

### 定位
- 全功能节点，4 cond 输出
- 高/低噪分离，精细控制
- 支持无损 latent 续接
- 叠加多种优秀特性（不使用显式模式切换）

### 输入

#### 核心连接（必须）

| 参数 | 类型 | 说明 |
|------|------|------|
| positive / negative | CONDITIONING | |
| vae | VAE | |

#### 节点控件（数值/开关）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| width / height | INT | 832 / 480 | |
| length | INT | 81 | 生成帧数 |
| motion_amplitude | FLOAT | 1.15 | 动作幅度增强 |
| motion_frames | INT | 5 | 续接重叠帧数 |
| correct_strength | FLOAT | 0.01 | 色彩校正强度 |
| color_protect | BOOLEAN | True | 启用色彩保护 |
| svi_compatible | BOOLEAN | False | 兼容 SVI 采样器 |

#### 可选连接

| 参数 | 类型 | 说明 |
|------|------|------|
| start_image | IMAGE | 首帧 |
| end_image | IMAGE | 尾帧 |
| clip_vision_start / end | CLIP_VISION_OUTPUT | |
| prev_latent | LATENT | 前置 latent（无损续接） |
| reference_latents | LATENT | 风格参考（任意数量） |
| reference_motion | LATENT | 动作参考（有输入即启用） |

### 输出

| 输出 | 类型 |
|------|------|
| high_positive / high_negative | CONDITIONING |
| low_positive / low_negative | CONDITIONING |
| latent | LATENT |

### 核心设计：锁定策略

**mask 共用，concat_latent 因 motion_amplitude 增强而分离。**

#### 锁定策略

| 区域 | 锁定方式 | mask 值 | 说明 |
|------|----------|---------|------|
| 首帧 (position 0) | **硬锁定** | 0.0 | prev_latent 覆盖 start_image |
| motion_latent (position 1~N) | **软锁定** | 1.0 | 仅注入 concat_latent，不锁定 mask |
| 尾帧 (position -1) | **硬锁定** | 0.0 | end_image 锁定 |
| 中间区域 | 生成 | 1.0 | 由模型自由生成 |

#### prev_latent 覆盖规则

```
if prev_latent is not None:
    # 覆盖模式：motion_latent 从 position 0 开始
    motion_latent = prev_latent["samples"][:, :, -motion_latent_count:]
    concat_latent[:, :, :motion_latent_count] = motion_latent
    mask[:, :, :1] = 0.0  # 只硬锁首帧
    # start_image 被忽略
else:
    # 首发模式：start_image 在 position 0
    if start_image is not None:
        image[0] = start_image[0]
        mask[:, :, :1] = 0.0  # 锁定首帧
```

#### 高/低噪分离机制

| 组件 | 高噪 | 低噪 | 说明 |
|------|------|------|------|
| mask | 共用 | 共用 | 首尾硬锁，中间软锁 |
| concat_latent | **增强版** | **原始版** | motion_amplitude + color_protect 仅高噪 |
| reference_latents | 自动生成 | 外部优先 | 无外部输入时复用高噪 |

#### concat_latent 分离流程

```
concat_latent_base = vae.encode(image)  # 基础编码
    ↓
注入 motion_latent (if prev_latent)
    ↓
concat_latent_original = clone()  ──────────────────→ 低噪使用
    ↓
apply_motion_amplitude() (if > 1.0)
    ↓
apply_color_protect() (if enabled)
    ↓
concat_latent_enhanced  ────────────────────────────→ 高噪使用
```

#### Reference Latent 逻辑

| 阶段 | reference_latents 来源 |
|------|------------------------|
| 高噪 | 自动生成：首帧 + prev_latent[-1] + end_image |
| 低噪 (有外部输入) | 仅使用外部 reference_latents |
| 低噪 (无外部输入) | 复用高噪的 reference |

### 来源整合

- **PainterI2VAdvanced**: motion_amplitude 增强 + color_protect
- **Wan22FMLF SVI**: prev_latent 无损续接 + motion_latent 软锁定

### 设计要点

1. **4 cond 输出**：需配合 PainterSamplerAdvanced
2. **高/低噪共用 concat_latent + mask**：简化逻辑
3. **首尾硬锁 + 中间软锁**：与 Wan22FMLF SVI 一致
4. **prev_latent 覆盖 start_image**：续接场景下忽略 start_image
5. **reference_latent 自动管理**：高噪从锚点图自动生成，低噪优先外部输入

---

## 配套采样器

| 采样器 | 适配节点 |
|--------|---------|
| PainterSampler | I2V, Extend (2 cond) |
| PainterSamplerAdvanced | Advanced (4 cond) |

---

## 下一步

1. ~~PainterI2V~~ ✅
2. ~~PainterI2V Extend~~ ✅ (AUTO_CONTINUE 已实现)
3. 🔄 PainterI2V Advanced 重构（特性叠加设计）
4. 测试 + 调优
