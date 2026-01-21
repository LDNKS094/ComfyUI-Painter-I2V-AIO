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
| color_protect | BOOLEAN | True | 色彩保护 |
| svi_mode | BOOLEAN | False | SVI LoRA 兼容模式 |

#### 可选连接

| 参数 | 类型 | 说明 |
|------|------|------|
| start_image | IMAGE | 首帧 |
| end_image | IMAGE | 尾帧 |
| clip_vision | CLIP_VISION_OUTPUT | 语义引导 |

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

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| width / height | INT | 832 / 480 | | |
| length | INT | 81 | | 生成帧数 |
| batch_size | INT | 1 | | |
| overlap_frames | INT | 4 | 4-8 | 重叠帧数（统一参数，控制不同模式行为） |
| svi_mode | BOOLEAN | False | | SVI 模式开关 |
| motion_amplitude | FLOAT | 1.15 | 1.0-2.0 | 动作幅度增强（仅 CONTINUITY 模式） |
| color_protect | BOOLEAN | True | | 启用色彩保护（仅 CONTINUITY 模式） |

#### 可选连接

| 参数 | 类型 | 说明 |
|------|------|------|
| anchor_image | IMAGE | SVI 锚点 + reference_latent 来源（两种模式通用） |
| end_image | IMAGE | 目标尾帧 |
| clip_vision | CLIP_VISION_OUTPUT | 语义引导 |

### 输出
`positive`, `negative`, `latent` (2 cond)

### 双模式设计

#### 模式切换
通过 `svi_mode` 布尔开关控制：

| svi_mode | 模式名称 | 用途 |
|----------|----------|------|
| False | **CONTINUITY** | 原生 I2V 动作接续（新发现） |
| True | **SVI** | SVI LoRA 长视频生成 |

#### 模式对比

| 特性 | **CONTINUITY (非 SVI)** | **SVI** |
|------|-------------------------|---------|
| **concat_latent 结构** | `[start, 灰色..., middle, 灰色...]` | `[anchor, motion, zero_padding...]` |
| **start/anchor 来源** | `previous_video[-overlap_frames]` | `anchor_image` 或 `previous_video[0]` |
| **middle/motion 来源** | `previous_video[-1]` 放在 `idx=overlap_frames` | 编码整个 previous_video 后取最后 N 帧 latent |
| **mask 策略** | start=锁定(0), middle=弱锁定 | anchor=锁定(0), motion=不锁定(1) |
| **middle_strength** | 自动计算: `overlap_frames * 0.025` | 不适用 |
| **padding 类型** | 灰色图像 encode | `latents_mean` (零值 latent) |
| **reference_latent** | `anchor_image` 或 `previous_video[-1]` | `anchor_image` 或 `previous_video[0]` |

### CONTINUITY 模式详解（新发现）

利用 FLF2V 的首中帧控制机制实现视频接续：

```
previous_video:  [...] [-N] [-N+1] ... [-2] [-1]
                        ↑                    ↑
                    start_image         middle_image

new_video:       [0]  [1]  ...  [N-1]  [N]  [N+1] ... [80]
                  ↑                     ↑
              start锁定            middle锚点
```

- `start = previous_video[-overlap_frames]`
- `middle = previous_video[-1]` 放在位置 `overlap_frames`
- 模型生成 start → middle（重叠区域）→ 新内容
- 后处理裁剪前 `overlap_frames` 帧实现无缝拼接

#### Middle Strength 自动计算

| overlap_frames | middle_strength | mask 值 |
|----------------|-----------------|---------|
| 4 | 0.10 | 0.90 |
| 5 | 0.125 | 0.875 |
| 6 | 0.15 | 0.85 |
| 8 | 0.20 | 0.80 |

公式: `middle_strength = overlap_frames * 0.025`

### SVI 模式详解

基于 SVI 2.0 Pro 设计：

```
concat_latent = [anchor_latent, motion_latent, zero_padding]
```

- `anchor_latent` = `anchor_image` 编码（或 `previous_video[0]`）
- `motion_latent` = 编码整个 previous_video 后取最后 N 帧 latent（见下方技术说明）
- `zero_padding` = `latents_mean` 填充

#### VAE Causal Temporal Encoding（关键发现）

VAE 是 causal temporal encoder，必须先编码整个视频再提取 latent：

```python
# ❌ 错误：只编码最后几帧
motion_latent = vae.encode(previous_video[-4:])

# ✅ 正确：编码整个视频，提取最后 N 个 latent
previous_encoded = vae.encode(previous_video)
motion_latent = previous_encoded[:, :, -context_latent_count:]
```

#### context_latent_count 参数

| 参数 | 基础节点 | Advanced 节点 |
|------|----------|---------------|
| 默认值 | 11（内部固定） | 11（可调节） |
| 行为 | 向下对齐（previous_video 帧数不足时自动减少） | 同左 |
| 公式 | `context_latent_count = min(11, (prev_frames - 1) // 4 + 1)` | 同左 |

帧数对应关系：
- 11 latent = 41 像素帧
- 6 latent = 21 像素帧
- 2 latent = 5 像素帧

### 内部行为

- **reference_latent 总是启用**：
  - 有 `anchor_image` → 使用 `anchor_image`
  - 无 `anchor_image` → 非 SVI 用 `previous_video[-1]`，SVI 用 `previous_video[0]`
- **motion_amplitude + color_protect**：仅非 SVI 模式生效
- **end_image**：两种模式都支持，放在序列末尾并锁定

### 来源
- 原 PainterLongVideo + Wan AUTO_CONTINUE
- 新增 Start-Middle Continuity 发现（2026-01-21）
- SVI 2.0 Pro 设计参考

---

## 节点 3: PainterI2V Advanced

### 定位
- 全功能节点，4 cond 输出
- 高/低噪分离，精细控制
- 支持无损 latent 续接（直接输入 `previous_latent`）
- 完全覆盖 PainterI2V 和 PainterI2VExtend 的所有场景

### 四种场景

| 场景 | svi_mode | previous_latent | 机制 |
|------|----------|-----------------|------|
| 标准初始生成 | False | None | 灰色填充 + start/end image 编码 |
| 标准视频延续 | False | 有 | motion latent 注入到 concat_latent 开头 |
| SVI 初始生成 | True | None | 零填充(latents_mean) + anchor 编码 |
| SVI 视频延续 | True | 有 | [anchor, motion, padding] 结构 |

**与 Extend 节点的区别**：Advanced 直接接收 `previous_latent`（已编码），无需 VAE encode。

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
| motion_amplitude | FLOAT | 1.15 | 动作幅度增强（仅高噪） |
| motion_latent_count | INT | 1 | 从 previous_latent 末端取多少帧 |
| correct_strength | FLOAT | 0.01 | 色彩校正强度 |
| color_protect | BOOLEAN | True | 启用色彩保护（仅高噪） |
| svi_mode | BOOLEAN | False | SVI LoRA 兼容模式 |

#### 可选连接

| 参数 | 类型 | 说明 |
|------|------|------|
| start_image | IMAGE | 首帧（被 previous_latent 覆盖） |
| end_image | IMAGE | 尾帧 |
| clip_vision | CLIP_VISION_OUTPUT | 语义引导 |
| previous_latent | LATENT | 前置 latent（无损续接） |
| reference_latent | LATENT | 外部风格参考（低噪优先使用） |

### 输出

| 输出 | 类型 |
|------|------|
| high_positive / high_negative | CONDITIONING |
| low_positive / low_negative | CONDITIONING |
| latent | LATENT |

### 核心设计

#### 1. concat_latent 构建

**标准模式 (svi_mode=False)**：
```python
# 初始生成：灰色填充 + 锚点图像
image = gray_fill(length)
image[0] = start_image
image[-1] = end_image
concat_latent = vae.encode(image)

# 视频延续：注入 motion latent
motion_latent = previous_latent[:, :, -motion_latent_count:]
concat_latent[:, :, :motion_latent_count] = motion_latent
```

**SVI 模式 (svi_mode=True)**：
```python
# 初始生成：零填充 + anchor
concat_latent = get_svi_padding_latent()  # latents_mean
concat_latent[:, :, :1] = start_latent

# 视频延续：[anchor, motion, padding]
anchor = previous_latent[:, :, :1]
motion = previous_latent[:, :, -motion_latent_count:]
concat_latent[:, :, :1] = anchor
concat_latent[:, :, 1:1+motion_latent_count] = motion
```

#### 2. mask 策略（高/低噪共用）

| 区域 | mask 值 | 说明 |
|------|---------|------|
| 首帧 | 0.0 | 硬锁定 |
| motion 区域 | 1.0 | 软锁定（仅 concat_latent 注入） |
| 尾帧 | 0.0 | 硬锁定（如有 end_image） |
| 其他 | 1.0 | 自由生成 |

#### 3. 高/低噪分离

| 组件 | 高噪 | 低噪 |
|------|------|------|
| concat_latent | 增强版（motion_amplitude + color_protect） | 原始版 |
| mask | 共用 | 共用 |
| reference_latent | 自动生成（start + end） | 外部优先，无则复用高噪 |

#### 4. 处理流程

```
concat_latent = build_based_on_mode()
    ↓
inject_motion_latent() (if previous_latent)
    ↓
concat_latent_original = clone()  ────────→ 低噪
    ↓
apply_motion_amplitude() (if > 1.0)
    ↓
apply_color_protect() (if enabled)
    ↓
concat_latent_enhanced  ──────────────────→ 高噪
```

### 来源整合

- **PainterI2VAdvanced**: motion_amplitude + color_protect + 高低噪分离
- **Wan22FMLF SVI**: previous_latent 无损续接
- **PainterI2VExtend**: 双模式设计思路

### 设计要点

1. **4 cond 输出**：需配合 PainterSamplerAdvanced
2. **previous_latent 覆盖 start_image**：续接场景下 start_image 被忽略
3. **无需 context_latent_count**：直接从 previous_latent 末端获取
4. **SVI 模式差异**：使用 [anchor, motion, padding] 结构

---

## 配套采样器

| 采样器 | 适配节点 |
|--------|---------|
| PainterSampler | I2V, Extend (2 cond) |
| PainterSamplerAdvanced | Advanced (4 cond) |

---

## 下一步

1. ~~PainterI2V~~ ✅
2. ~~PainterI2V Extend~~ ✅ (双模式已实现)
3. ~~PainterI2V Advanced~~ ✅ (四场景 + 高低噪分离)
4. 测试 + 调优
