[English](README.md) | [中文](README_zh.md)

# ComfyUI-PainterAIO

Painter 系列 ComfyUI 视频生成节点整合包，支持 Wan 2.1/2.2。

本项目整合并重构了多个 ComfyUI 视频节点项目，统一 API、优化代码结构。

---

## 来源项目

| 原项目 | 作者 | 功能 |
|--------|------|------|
| [ComfyUI-PainterI2V](https://github.com/princepainter/ComfyUI-PainterI2V) | princepainter | 图生视频动态增强 |
| [ComfyUI-PainterI2Vadvanced](https://github.com/princepainter/ComfyUI-PainterI2Vadvanced) | princepainter | 双采样器 + 颜色保护 |
| [ComfyUI-PainterLongVideo](https://github.com/princepainter/ComfyUI-PainterLongVideo) | princepainter | 长视频接续 |
| [Comfyui-PainterFLF2V](https://github.com/princepainter/Comfyui-PainterFLF2V) | princepainter | 首尾帧插值 |
| [ComfyUI-Wan22FMLF](https://github.com/wallen0322/ComfyUI-Wan22FMLF) | wallen0322 | 多帧参考条件控制 |

---

## 节点列表

| 节点名称 | 功能 |
|----------|------|
| **PainterI2V** | 图生视频，支持 I2V / FLF2V 模式，动态增强 |
| **PainterI2VAdvanced** | 双阶段采样（高/低噪分离），支持循环接续 |
| **PainterI2VExtend** | 视频接续，长视频生成 |
| **PainterSampler** | 采样器，支持动态增强 |
| **PainterSamplerAdvanced** | 双阶段采样器，配合 PainterI2VAdvanced 使用 |

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LDNKS094/ComfyUI-PainterAIO.git
```

---

## 主要特性

- **动态增强**: 修复 4-step LoRA（如 lightx2v）的慢动作问题
- **颜色保护**: 防止动态增强后的偏绿、偏暗
- **双阶段采样**: 高噪阶段动态增强，低噪阶段语义参考（仅 Advanced）
- **首尾帧插值**: FLF2V 模式，支持起始帧和结束帧锚定
- **长视频接续**: 无缝衔接上一段视频
- **SVI LoRA 兼容**: 支持 SVI 模式的 latents_mean 填充
- **循环支持**: 自动在 latent/image 间转换，适配 ComfyUI 循环工作流

---

## 节点详解

### PainterI2V

基础图生视频节点，支持动态增强。

| 参数 | 说明 |
|------|------|
| motion_amplitude | 4-step LoRA 修复。1.1-1.2 普通，1.2-1.5 快动作 |
| color_protect | 防止动态增强后颜色漂移 |
| svi_mode | SVI LoRA 模式，使用 latents_mean 填充 |
| start_image | 起始帧参考 |
| end_image | 结束帧（FLF2V 模式） |
| clip_vision | 语义引导 |

### PainterI2VAdvanced

高级节点，支持双阶段采样（高/低噪分离）。

**输出 4 个条件**: `high_positive`, `high_negative`, `low_positive`, `low_negative`

需配合 **PainterSamplerAdvanced** 使用。

#### 高噪阶段（动态）
- `motion_amplitude` - 动态增强强度
- `color_protect` - 颜色保护
- `end_image` - 结束帧锚定

#### 低噪阶段（语义）
- `reference_latent` - 来自 start_image（自动）
- `clip_vision` - 语义引导
- `correct_strength` - 参考修正强度（推荐 0.01-0.05）

#### 接续参数（仅标准模式）
- `overlap_frames` - 重叠帧数（推荐 4-8）
- `continuity_strength` - 动作锁定强度（推荐 0.1-0.2）
- `previous_latent` 或 `previous_image` - 上一段用于接续

#### 模式差异

| 特性 | 标准模式 | SVI 模式 |
|------|----------|----------|
| 接续来源 | previous_image | previous_latent |
| overlap_frames | 使用（4-8） | 固定（1 latent 帧） |
| concat 填充 | 灰色（0.5）编码 | latents_mean（零） |

### PainterI2VExtend

视频接续节点，用于长视频生成。

| 参数 | 说明 |
|------|------|
| overlap_frames | 连续性重叠帧数（推荐 4-8，仅标准模式） |
| motion_amplitude | 4-step LoRA 修复（两种模式均生效） |
| color_protect | 颜色保护（两种模式均生效） |
| svi_mode | SVI LoRA 模式，使用 anchor + 最后一帧 latent |
| previous_video | 需要接续的上一段视频 |
| anchor_image | 锚定帧（默认使用 previous_video[0]） |
| end_image | 目标结束帧 |
| clip_vision | 语义引导 |

---

## 参数建议

### motion_amplitude（动态增强幅度）

| 运动类型 | 推荐值 |
|----------|--------|
| 轻微动作（说话、眨眼） | 1.0 - 1.1 |
| 中等动作（走路、手势） | 1.1 - 1.2 |
| 大幅度动作（跑步、跳跃） | 1.2 - 1.5 |

### color_protect（颜色保护）

建议保持开启（默认 True），可有效防止动态增强后的颜色漂移。

### overlap_frames（Advanced/Extend）

用于接续的像素帧重叠数量。内部会转换为 latent 索引（`overlap_frames // 4`）。

| 用途 | 推荐值 |
|------|--------|
| 普通接续 | 4 |
| 更平滑过渡 | 8 |

### continuity_strength（仅 Advanced）

控制片段间的动作锁定强度。

| 用途 | 推荐值 |
|------|--------|
| 普通接续 | 0.1 |
| 更强动作锁定 | 0.2 |

### correct_strength（仅 Advanced）

低噪阶段的参考 latent 修正强度。

| 用途 | 推荐值 |
|------|--------|
| 普通 | 0.01 - 0.03 |
| 更强参考 | 0.03 - 0.05 |

---

## 致谢

- **[princepainter](https://github.com/princepainter)**
- **[wallen0322](https://github.com/wallen0322)**
- **Wan2.1/2.2 团队**
- **ComfyUI 社区**

---

## 许可证

MIT License
