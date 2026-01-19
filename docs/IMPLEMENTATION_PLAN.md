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

| 参数 | 类型 | 说明 |
|------|------|------|
| positive / negative | CONDITIONING | |
| vae | VAE | |
| width / height / length | INT | 默认 832×480, 81帧 |
| motion_amplitude | FLOAT | 默认 1.15 |
| start_image | IMAGE | 首帧 (可选) |
| end_image | IMAGE | 尾帧 (可选) |
| reference_video | IMAGE | 动作参考 (可选) |
| clip_vision_start / end | CLIP_VISION_OUTPUT | (可选) |
| enable_reference_latent | BOOL | 默认 True |
| svi_compatible | BOOL | 默认 False |

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

| 参数 | 类型 | 说明 |
|------|------|------|
| positive / negative | CONDITIONING | |
| vae | VAE | |
| width / height / length | INT | 默认 832×480, 81帧 |
| motion_amplitude | FLOAT | 默认 1.15 |
| previous_video | IMAGE | **必须** - 前置视频 |
| motion_frames | INT | 重叠帧数，默认 5 |
| end_image | IMAGE | 目标尾帧 (可选) |
| reference_video | IMAGE | 动作参考 (可选) |
| clip_vision_start / end | CLIP_VISION_OUTPUT | (可选) |
| enable_reference_latent | BOOL | 默认 True |
| svi_compatible | BOOL | 默认 False |

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

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| positive / negative | CONDITIONING | ✅ | |
| vae | VAE | ✅ | |
| width / height / length | INT | ✅ | |
| motion_amplitude | FLOAT | ✅ | 默认 1.3 |
| color_protect | BOOLEAN | ❌ | 默认 True |
| correct_strength | FLOAT | ❌ | 默认 0.01 |
| start_image | IMAGE | ❌ | 首帧 |
| end_image | IMAGE | ❌ | 尾帧 |
| clip_vision_start / end | CLIP_VISION_OUTPUT | ❌ | |
| prev_latent | LATENT | ❌ | 前置 latent（无损续接） |
| motion_frames | INT | ❌ | 续接帧数，默认 5 |
| reference_latents | LATENT | ❌ | 风格参考（任意数量） |
| reference_motion | LATENT | ❌ | 动作参考 |
| enable_reference_latent | BOOLEAN | ❌ | 默认 True |
| enable_reference_motion | BOOLEAN | ❌ | 默认 True |
| svi_compatible | BOOLEAN | ❌ | 默认 False |

### 输出

| 输出 | 类型 |
|------|------|
| high_positive / high_negative | CONDITIONING |
| low_positive / low_negative | CONDITIONING |
| latent | LATENT |

### 核心设计：特性叠加

根据输入自然叠加特性，不使用显式模式判断：

#### 高噪 conditioning
- 包含所有锚点：start_image + motion_latent (from prev_latent) + end_image
- 应用 motion_amplitude 增强 + color_protect
- mask 锁定所有锚点区域

#### 低噪 conditioning
- 最小锚点：start_image + end_image（不含 motion_latent）
- 使用原始版本（不应用 motion_amplitude）
- mask 只锁定 start/end

#### 特性叠加表

| 输入 | 高噪 concat_latent | 高噪 mask | 低噪 concat_latent | 低噪 mask |
|------|-------------------|-----------|-------------------|-----------|
| start_image | ✅ 包含 | 锁定 | ✅ 包含 | 锁定 |
| end_image | ✅ 包含 | 锁定 | ✅ 包含 | 锁定 |
| prev_latent + motion_frames | ✅ 注入 motion_latent | 锁定 | ❌ 不包含 | 不锁定 |
| motion_amplitude > 1.0 | ✅ 增强 | - | ❌ 原始版 | - |
| color_protect | ✅ 应用 | - | ❌ 不应用 | - |

### 来源整合

- **PainterI2VAdvanced**: motion_amplitude 增强 + color_protect + 高/低 latent 版本分离
- **Wan22FMLF SVI**: prev_latent 无损续接 + 高/低 mask 区域分离

### 设计要点

1. **4 cond 输出**：需配合 PainterSamplerAdvanced
2. **特性自然叠加**：有什么输入就应用什么特性
3. **两种分离机制共存**：latent 增强分离 + mask 区域分离
4. **color_protect 独立**：与续接机制互不影响
5. **reference_latents 任意数量**：自动合并内部 + 外部输入

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
