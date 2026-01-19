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
- 单段视频首发生成
- 无前置视频输入
- 入门级，低理解成本

### 输入

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| positive | CONDITIONING | ✅ | |
| negative | CONDITIONING | ✅ | |
| vae | VAE | ✅ | |
| width | INT | ✅ | 默认 832 |
| height | INT | ✅ | 默认 480 |
| length | INT | ✅ | 默认 81 |
| motion_amplitude | FLOAT | ✅ | 默认 1.15 |
| start_image | IMAGE | ❌ | 首帧 |
| end_image | IMAGE | ❌ | 尾帧 |
| reference_video | IMAGE | ❌ | 动作参考（提取 reference_motion） |
| clip_vision_start | CLIP_VISION_OUTPUT | ❌ | |
| clip_vision_end | CLIP_VISION_OUTPUT | ❌ | |
| enable_reference_latent | BOOL | ❌ | 测试用开关 |
| svi_compatible | BOOL | ❌ | SVI LoRA 兼容模式，默认 False |

### 输出

| 输出 | 类型 |
|------|------|
| positive | CONDITIONING |
| negative | CONDITIONING |
| latent | LATENT |

### 模式自动切换

| 输入组合 | 模式 | 行为 |
|---------|------|------|
| 无图像 | T2V | 纯文本生成 |
| start_image | I2V | 首帧锚定 + reference_latents |
| start + end | FLF2V | 首尾帧锚定 + 频率分离增强 |
| + reference_video | 任意模式 + 动作引导 | 额外注入 reference_motion |

### 核心机制

- concat_latent_image + mask（首/尾帧硬锚定）
- reference_latents（I2V 模式，首帧风格引导）
- reference_motion（从 reference_video 提取，软引导整个视频动作）
- 频率分离运动增强（FLF2V 模式）
- 双 CLIP Vision concat（FLF2V 模式）
- ~~**空间梯度增强**（FLF2V 模式隐式启用，无开关）~~ **已放弃**
- **SVI 兼容模式**（`svi_compatible` 开关控制 latent 填充方式）

### reference_motion 机制

```python
# 仅当 reference_video 存在时提取
if reference_video is not None:
    # 提取 length 帧，匹配输出视频长度
    latent_frames = ((length - 1) // 4) + 1
    frames_to_extract = min(length, reference_video.shape[0])
    ref_motion = reference_video[-frames_to_extract:]
    
    # 不足则用灰帧填充前面
    if ref_motion.shape[0] < length:
        gray_fill = torch.ones([length, H, W, 3]) * 0.5
        gray_fill[-ref_motion.shape[0]:] = ref_motion
        ref_motion = gray_fill
    
    # VAE 编码 → reference_motion
    ref_motion_latent = vae.encode(ref_motion)  # [1, 16, latent_frames, H, W]
    conditioning["reference_motion"] = ref_motion_latent
```

### 来源

- 原 PainterI2V
- 原 PainterFLF2V（已合并）

---

## 节点 2: PainterI2V Extend

### 定位
- 视频续接专用
- 接收前置视频 (IMAGE)
- 入门级，与 PainterI2V 配合使用

### 输入

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| positive | CONDITIONING | ✅ | |
| negative | CONDITIONING | ✅ | |
| vae | VAE | ✅ | |
| width | INT | ✅ | 默认 832 |
| height | INT | ✅ | 默认 480 |
| length | INT | ✅ | 默认 81 |
| motion_amplitude | FLOAT | ✅ | 默认 1.15 |
| previous_video | IMAGE | ✅ | 前置视频（核心输入） |
| motion_frames | INT | ❌ | 接续重叠帧数，默认 5 |
| end_image | IMAGE | ❌ | 目标尾帧 |
| reference_video | IMAGE | ❌ | 动作参考（提取 reference_motion） |
| clip_vision | CLIP_VISION_OUTPUT | ❌ | |
| enable_reference_latent | BOOL | ❌ | 测试用开关 |
| svi_compatible | BOOL | ❌ | SVI LoRA 兼容模式，默认 False |

> **注意**：无 `initial_reference` 输入。风格控制隐式使用 `previous_video` 首尾帧。

### 输出

| 输出 | 类型 |
|------|------|
| positive | CONDITIONING |
| negative | CONDITIONING |
| latent | LATENT |

### 两个核心机制

#### 1. 动作接续（motion_frames）— AUTO_CONTINUE 模式

> 借鉴 Wan22FMLF 的 AUTO_CONTINUE 模式，使用 IMAGE 空间操作

```
previous_video[-motion_frames:] → 填入输出序列开头 → mask=0 硬锁定
```

- 目的：保证动作在接续点平滑过渡
- 输出包含这些重叠帧
- 实际新生成帧数 = length - motion_frames
- 后处理可裁剪重叠帧（使用 KJNodes ImageBatchExtendWithOverlap 等）
- **高噪/低噪阶段使用相同 conditioning**（与 SVI 模式的关键区别）

**实现逻辑**：
```python
# 1. 从 previous_video 提取最后 motion_frames 帧
overlap_frames = previous_video[-motion_frames:]

# 2. 填入输出序列开头
image = torch.ones((length, height, width, 3), device=device) * 0.5
image[:motion_frames] = overlap_frames

# 3. 其余帧填充（如 end_image）
if end_image is not None:
    image[-1:] = end_image

# 4. VAE 编码
concat_latent_image = vae.encode(image)

# 5. 创建 mask，锁定 motion_frames 区域
motion_latent_frames = ((motion_frames - 1) // 4) + 1
mask[:, :, :motion_latent_frames * 4] = 0.0  # 硬锁定
```

#### 2. 动作引导（reference_motion）

```
reference_video[-length:] → VAE encode → reference_motion（软引导整个视频）
```

- **仅从 reference_video 提取，不从 previous_video 隐式提取**
- 避免用户只接 previous_video 时出现重复动作
- 若需要动作引导，用户需显式连接 reference_video

### 模式自动切换

| 输入组合 | 模式 | 行为 |
|---------|------|------|
| previous_video | 续接 | 取末 motion_frames 帧重叠 + 末帧锚定 |
| previous_video + end_image | 续接+尾帧 | 首尾锚定 |
| + reference_video | 任意模式 + 动作引导 | 额外注入 reference_motion |

### reference_latents 构成（隐式）

```python
# Extend 节点自动构建，用户无需关心
ref_latents = []
# 使用 previous_video 首尾帧作为风格参考
ref_latents.append(vae.encode(previous_video[-1:]))  # 末帧
# 如需首帧：ref_latents.append(vae.encode(previous_video[:1]))
```

### 设计要点

1. **必须有 previous_video**：与 PainterI2V 的核心区别
2. **motion_frames**：控制重叠帧数，借鉴 Wan AUTO_CONTINUE
3. **reference_motion 显式提取**：仅从 reference_video，不隐式提取
4. **无 initial_reference**：简化接口，风格控制隐式使用首尾帧
5. **简单输出**：2 cond，与普通采样器配合

### 来源

- 原 PainterLongVideo（重构）
- Wan AUTO_CONTINUE（motion_frames 机制）

---

## 节点 3: PainterI2V Advanced

### 定位
- 全功能节点
- 支持所有场景
- 4 cond 输出，精细控制
- 支持无损 latent 续接
- **不进行隐式 VAE encode/decode**
- 高级用户使用

### 输入

| 参数 | 类型 | 必须 | 说明 |
|------|------|------|------|
| positive | CONDITIONING | ✅ | |
| negative | CONDITIONING | ✅ | |
| vae | VAE | ✅ | 仅用于 concat_latent 编码 |
| width | INT | ✅ | 默认 832 |
| height | INT | ✅ | 默认 480 |
| length | INT | ✅ | 默认 81 |
| motion_amplitude | FLOAT | ✅ | 1.0-2.0，默认 1.15 |
| color_protect | BOOLEAN | ❌ | 色彩保护，默认 True |
| correct_strength | FLOAT | ❌ | 色彩修正强度 |
| start_image | IMAGE | ❌ | 首帧（需 VAE 编码） |
| end_image | IMAGE | ❌ | 尾帧（需 VAE 编码） |
| clip_vision_start | CLIP_VISION_OUTPUT | ❌ | |
| clip_vision_end | CLIP_VISION_OUTPUT | ❌ | |
| prev_latent | LATENT | ❌ | 前置 latent（无损续接） |
| motion_frames | INT | ❌ | 接续重叠帧数 |
| reference_latents | LATENT | ❌ | 风格参考 latent（任意数量，用户自行 VAE encode） |
| reference_motion | LATENT | ❌ | 动作参考 latent（用户自行准备） |
| enable_reference_latent | BOOLEAN | ❌ | 启用 reference_latents |
| enable_reference_motion | BOOLEAN | ❌ | 启用 reference_motion |
| ~~spatial_gradient~~ | ~~BOOLEAN~~ | ~~❌~~ | ~~空间梯度增强（FLF2V 模式），默认 True~~ **已放弃** |
| svi_compatible | BOOLEAN | ❌ | SVI LoRA 兼容模式，默认 False |

> **注意**：`reference_latents` 和 `reference_motion` 均为 LATENT 类型，用户需使用外部 VAE Encode 节点准备。
> 节点不进行隐式 VAE encode/decode，保持完全控制权。

### 输出

| 输出 | 类型 |
|------|------|
| high_positive | CONDITIONING |
| high_negative | CONDITIONING |
| low_positive | CONDITIONING |
| low_negative | CONDITIONING |
| latent | LATENT |

### 模式自动切换

| 输入组合 | 模式 | 行为 |
|---------|------|------|
| 无图像 | T2V | 纯文本 |
| start_image | I2V | 首帧锚定 |
| start + end | FLF2V | 首尾帧锚定 |
| prev_latent | LATENT_CONTINUE | 无损续接（简单模式） |
| prev_latent + start_image | **SVI_CONTINUE** | 高/低噪分离续接 |

### SVI_CONTINUE 模式（高级续接）

> 借鉴 Wan22FMLF 的 SVI 模式，实现高/低噪阶段分离的精细控制

**触发条件**：`prev_latent` + `start_image` + `motion_frames > 0`

**核心机制**：
```python
# 高噪阶段：用 prev_latent 的最后 N latent frames 保证动作连续
motion_latent_count = ((motion_frames - 1) // 4) + 1
motion_latent = prev_latent["samples"][:, :, -motion_latent_count:]

# 构建高噪 conditioning
image_cond_latent = torch.zeros(...)
image_cond_latent[:, :, :1] = anchor_latent  # start_image 编码
image_cond_latent[:, :, 1:1+motion_latent_count] = motion_latent  # 续接帧
# ... 可选 middle_image, end_image

positive_high = set_values(positive, {
    "concat_latent_image": image_cond_latent,
    "concat_mask": mask_high  # 锁定 anchor + motion 区域
})

# 低噪阶段：只用 start_image 保证风格一致性，不受 motion_latent 约束
image_low = gray_canvas
image_low[:start_image.shape[0]] = start_image
concat_latent_image_low = vae.encode(image_low)

positive_low = set_values(positive, {
    "concat_latent_image": concat_latent_image_low,
    "concat_mask": mask_low  # 只锁定 start_image 区域
})
```

**优势**：
- 高噪阶段：motion_latent 硬锁定，保证帧间连续性
- 低噪阶段：不受 motion_latent 约束，避免颜色/风格漂移
- 无损 latent 空间操作，跳过 VAE 编解码

### 核心机制

- 4 cond 输出（高噪/低噪 分离）
- 双版本 latent（增强版/原始版）
- 色彩保护算法
- reference_latents（LATENT 输入，任意数量）
- reference_motion（LATENT 输入）
- 无损 latent 续接
- 频率分离运动增强
- **空间梯度增强**（`spatial_gradient` 开关控制，默认开启，仅 FLF2V 模式生效）
- **无隐式 VAE encode/decode**

### 来源

- 原 PainterI2VAdvanced
- 原 PainterLongVideo（reference_motion 机制）
- Wan SVI（无损续接思路）
- Wan 空间梯度增强（`spatial_gradient` 开关控制）

### 设计要点

1. **4 cond 输出**：需配合 PainterSamplerAdvanced
2. **无损续接**：prev_latent 输入，跳过 VAE
3. **可选开关**：reference_latent、reference_motion 可独立控制
4. **色彩保护**：通道漂移检测 + 亮度保护
5. **无隐式 VAE**：reference_latents 和 reference_motion 均需用户自行编码
6. **任意数量参考**：reference_latents 接受任意数量 latent（列表形式）

### reference_latents 使用方式

```
[VAE Encode] → latent_1 ─┐
[VAE Encode] → latent_2 ─┼→ [Latent Batch] → reference_latents → [PainterI2V Advanced]
[VAE Encode] → latent_3 ─┘
```

用户可自由组合任意数量的参考帧，节点直接使用而不做额外处理。

---

## 配套采样器

| 采样器 | 输入 | 适配节点 |
|--------|------|---------|
| PainterSampler | 2 cond (pos/neg) | PainterI2V, PainterI2V Extend |
| PainterSamplerAdvanced | 4 cond (high/low × pos/neg) | PainterI2V Advanced |

---

## 工作流示例

### 简单单段生成

```
[PainterI2V] → [PainterSampler] → VIDEO
```

### 简单长视频

```
[PainterI2V] → [Sampler] → video_1
                              ↓
[PainterI2V Extend] ←─ previous_video
        ↓
   [Sampler] → video_2 (包含 motion_frames 重叠帧)
        ↓
   [裁剪重叠帧] → video_2_trimmed
        ↓
   [拼接] → 长视频
```

### 动作参考生成

```
[Load Video] → reference_video
                    ↓
[PainterI2V] ←─ reference_video + start_image
        ↓
   [Sampler] → VIDEO (动作风格类似参考视频)
```

### 高级长视频（无损 + 多参考）

```
[Load Image] → [VAE Encode] → ref_latent_1 ─┐
[Load Image] → [VAE Encode] → ref_latent_2 ─┼→ [Latent Batch] → reference_latents
[Load Image] → [VAE Encode] → ref_latent_3 ─┘
                                                        ↓
[PainterI2V Advanced] ← reference_latents + prev_latent
         ↓
  [SamplerAdvanced] → latent_1
         ↓
     [Loop...]
         ↓
  [VAE Decode] → FINAL_VIDEO
```

---

## 技术整合清单

### 从 Painter 保留

| 技术 | 应用节点 |
|------|---------|
| 频率分离运动增强 | I2V (FLF2V), Advanced |
| reference_latents | I2V, Extend (隐式), Advanced (显式 LATENT) |
| reference_motion | I2V, Extend (从 reference_video), Advanced (显式 LATENT) |
| 双 CLIP Vision concat | I2V (FLF2V), Advanced |
| 4 cond 分离 | Advanced |
| 色彩保护算法 | Advanced |

### 从 Wan 借鉴

| 技术 | 应用节点 | 说明 |
|------|---------|------|
| AUTO_CONTINUE 模式 | Extend | IMAGE 空间，2 cond，高/低噪相同 |
| SVI_CONTINUE 模式 | Advanced | LATENT 空间，4 cond，高/低噪分离 |
| LATENT_CONTINUE 模式 | Advanced | 简单无损续接，1 latent frame |
| 无损 latent 续接 | Advanced | 跳过 VAE 编解码 |
| 双 mask 系统 | Advanced | 高噪/低噪使用不同 mask |
| ~~空间梯度增强~~ | ~~I2V (FLF2V 隐式), Advanced (开关控制)~~ | **已放弃** |
| SVI 兼容模式 | I2V, Extend, Advanced | latent 填充方式（开关控制） |

### 节点对比

| 特性 | PainterI2V | Extend | Advanced |
|------|-----------|--------|----------|
| 隐式 VAE encode | ✅ | ✅ | ❌ |
| reference_latents 类型 | 隐式 | 隐式 | LATENT |
| reference_motion 类型 | 从 IMAGE | 从 IMAGE | LATENT |
| 参考数量 | 1 | 1 | 任意 |
| 输出 | 2 cond | 2 cond | 4 cond |
| 续接模式 | - | AUTO_CONTINUE | SVI_CONTINUE / LATENT_CONTINUE |
| 续接输入 | - | IMAGE (previous_video) | LATENT (prev_latent) |
| 高/低噪分离 | ❌ | ❌ | ✅ |
| SVI 兼容填充 | 开关 | 开关 | 开关 |

---

## 关键公式

### Image frames ↔ Latent frames 转换

```python
# image → latent
latent_frames = ((image_frames - 1) // 4) + 1

# latent → image
image_frames = (latent_frames - 1) * 4 + 1
```

| image frames | latent frames |
|--------------|---------------|
| 49 | 13 |
| 73 | 19 |
| 81 | 21 |
| 97 | 25 |

---

## 视频续接模式深度分析

> 基于 Wan22FMLF 代码分析的新发现

### 三种续接模式对比

| 模式 | 输入类型 | continue_frames_count 用途 | 高噪 mask | 低噪 mask | cond 输出 |
|------|----------|---------------------------|-----------|-----------|-----------|
| **AUTO_CONTINUE** | `motion_frames` (IMAGE) | 控制取多少 image frames | 锁定 motion_frames 区域 | 锁定 motion_frames 区域 | 2 cond 可用 |
| **SVI** | `prev_latent` (LATENT) | 转换为 latent frames | 锁定 motion_latent 区域 | **不锁定**，用 start_image | 需要 3+ cond |
| **LATENT_CONTINUE** | `prev_latent` (LATENT) | 仅判断是否启用 | 锁定 1 latent frame | 锁定 1 latent frame | 2 cond 可用 |

### SVI 模式的特殊性（`svi_continue_mode`）

**触发条件**：
```python
if long_video_mode == 'SVI' and has_prev_latent and continue_frames_count > 0:
    svi_continue_mode = True
```

**关键差异**：SVI 模式下高噪/低噪阶段使用不同的 conditioning：

| 阶段 | concat_latent_image | mask 锁定区域 | 目的 |
|------|---------------------|--------------|------|
| **高噪 (high)** | 包含 motion_latent | motion_latent 区域锁定 | 保证动作连续性 |
| **低噪 (low)** | 只有 start_image | 只锁定 start_image 区域 | 保证风格/颜色一致性 |

```python
# SVI 模式下低噪阶段的特殊处理
if svi_continue_mode:
    image_low = torch.ones((length, height, width, 3), device=device) * 0.5
    if start_image is not None:
        image_low[:start_image.shape[0]] = start_image[:, :, :, :3]
        mask_low_noise[:, :, :start_latent_frames * 4] = 0.0
    concat_latent_image_low = vae.encode(image_low[:, :, :, :3])
```

### 后处理拼接 vs Conditioning 层面的重叠

**重要发现**：这是两个不同的概念！

| 层面 | 参数 | 作用 |
|------|------|------|
| **Conditioning** | `continue_frames_count` / `motion_frames` | 告诉模型"接续这些帧" |
| **后处理拼接** | KJNodes `overlap` | 用于平滑混合两段解码后的视频 |

**这两个值可以不同**：
- SVI 模式下，conditioning 可能只需要少量帧（1 latent frame，即 continue_frames_count < 4）
- 但后处理拼接可能需要更多帧（如 overlap = 5）做平滑过渡

### 对节点设计的影响

#### PainterI2V Extend（2 cond 输出）

使用 **AUTO_CONTINUE** 模式：
- 高噪/低噪阶段使用相同的 conditioning
- `motion_frames` 同时控制：
  1. 从 previous_video 取多少帧
  2. 填入输出序列开头的帧数
  3. 后处理需要裁剪的帧数
- 简单直观，适合入门用户

#### PainterI2V Advanced（4 cond 输出）

可选择实现 **SVI 模式**：
- 高噪/低噪阶段使用不同的 conditioning
- 需要 `prev_latent` (LATENT) 输入实现无损续接
- 高噪阶段用 motion_latent 保证连续性
- 低噪阶段用 start_image 保证风格一致性
- 更复杂但效果可能更好

### 设计决策

1. **PainterI2V Extend**：仅实现 AUTO_CONTINUE 模式（IMAGE 空间，2 cond）
2. **PainterI2V Advanced**：实现 SVI 模式（LATENT 空间，4 cond，高/低噪分离）

---

## 待确认问题

1. **Extend 是否需要 motion_amplitude？** ✅ 已确认
   - 需要，默认 1.15
   
2. **Advanced 的 output_latent 是否需要？** ✅ 已确认
   - 不需要单独输出，复用 sampler 输出即可

3. **空间梯度增强是否加入？** ✅ 已确认
   - **PainterI2V**: FLF2V 模式（首尾帧都接入时）隐式启用，无开关
   - **PainterI2V Advanced**: 添加 `spatial_gradient` 开关，默认开启，用户可关闭

4. **命名确认** ✅ 已确认
   - 使用 PainterI2V Extend

5. **reference_motion 提取策略** ✅ 已确认
   - 仅从 reference_video 显式提取
   - 不从 previous_video 隐式提取

6. **分辨率默认值** ✅ 已确认
   - 统一使用 Wan 模型推荐值：832×480

---

## 技术规范

### API 统一

**所有节点使用新 ComfyExtension API**：
- `io.ComfyNode` 基类
- `define_schema()` 定义输入输出
- `execute()` 类方法

### 公共函数提取

将各节点可复用的逻辑提取到 `modules/common/utils.py`：

```python
# modules/common/utils.py

def encode_image_sequence(vae, images, width, height, length):
    """统一的图像序列编码"""
    pass

def create_concat_mask(latent_frames, anchor_positions, spacial_scale):
    """创建 concat_mask，支持多锚点"""
    pass

def apply_motion_amplitude(concat_latent, base_idx, amplitude, protect_brightness=True):
    """运动幅度增强（亮度保护）"""
    pass

def apply_frequency_separation(latent, linear_baseline, boost_scale):
    """频率分离增强（FLF2V 用）"""
    pass

def extract_reference_motion(vae, video_frames, width, height, target_length):
    """从视频提取 reference_motion latent，匹配目标长度"""
    pass

def build_reference_latents(vae, images_list, width, height):
    """构建 reference_latents 列表"""
    pass

def inject_conditioning(cond, values_dict, append_keys=None):
    """统一的 conditioning 注入"""
    pass

def merge_clip_vision_outputs(*outputs):
    """合并多个 CLIP Vision 输出"""
    pass
```

### 节点依赖关系

```
modules/common/utils.py          ← 公共函数
    ↑
modules/painteri2v/nodes.py      ← PainterI2V
modules/painteri2v_extend/nodes.py   ← PainterI2V Extend (新建)
modules/painteri2v_advanced/nodes.py ← PainterI2V Advanced (重构)
```

---

## 下一步

1. ~~确认以上规划~~ ✅
2. ~~开始实现 PainterI2V~~ ✅ 已完成
3. 创建 `modules/common/utils.py` 提取公共函数
4. 实现 PainterI2V Extend（基于 PainterLongVideo 重构）
5. 实现 PainterI2V Advanced（最复杂）
6. 测试 + 调优
