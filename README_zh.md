[English](README.md) | [中文](readme_zh.md)

# ComfyUI-PainterAIO

All-in-One collection of Painter's ComfyUI nodes for Wan 2.2 video generation.

本项目整合并重构了多个 ComfyUI 视频节点项目，统一 API、优化代码结构。

---

## 来源项目 / Based On

本项目基于以下开源项目修改和整合：

| 原项目 | 作者 | 功能 |
|--------|------|------|
| [ComfyUI-PainterI2V](https://github.com/princepainter/ComfyUI-PainterI2V) | princepainter | 图生视频动态增强 |
| [ComfyUI-PainterI2Vadvanced](https://github.com/princepainter/ComfyUI-PainterI2Vadvanced) | princepainter | 双采样器 + 颜色保护 |
| [ComfyUI-PainterLongVideo](https://github.com/princepainter/ComfyUI-PainterLongVideo) | princepainter | 长视频接续 |
| [Comfyui-PainterFLF2V](https://github.com/princepainter/Comfyui-PainterFLF2V) | princepainter | 首尾帧插值 |
| [ComfyUI-Wan22FMLF](https://github.com/wallen0322/ComfyUI-Wan22FMLF) | wallen0322 | 多帧参考条件控制 |

---

## 节点列表 / Nodes

| 节点名称 | 功能 |
|----------|------|
| **PainterI2V** | 图生视频，支持 I2V / FLF2V 模式，动态增强 |
| **PainterI2VAdvanced** | 高级版，双采样器输出，颜色保护 |
| **PainterI2VExtend** | 视频接续，长视频生成 |
| **PainterSampler** | 采样器，支持动态增强 |
| **PainterSamplerAdvanced** | 高级采样器，双采样器输出 |

---

## 安装 / Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LDNKS094/ComfyUI-Painter-I2V-AIO.git
```

---

## 主要特性 / Features

- **动态增强**: 修复 4-step LoRA（如 lightx2v）的慢动作问题
- **颜色保护**: 防止动态增强后的偏绿、偏暗
- **首尾帧插值**: FLF2V 模式，支持起始帧和结束帧锚定
- **长视频接续**: 无缝衔接上一段视频
- **SVI LoRA 兼容**: 支持 SVI 模式的 latents_mean 填充
- **双 CLIP Vision**: 支持起始和结束帧的语义引导

---

## 参数建议 / Parameters

### motion_amplitude（动态增强幅度）

| 运动类型 | 推荐值 |
|----------|--------|
| 轻微动作（说话、眨眼） | 1.0 - 1.1 |
| 中等动作（走路、手势） | 1.1 - 1.2 |
| 大幅度动作（跑步、跳跃） | 1.2 - 1.5 |

### color_protect（颜色保护）

建议保持开启（默认 True），可有效防止动态增强后的颜色漂移。

---

## 致谢 / Acknowledgements

- **[princepainter](https://github.com/princepainter)**
- **[wallen0322](https://github.com/wallen0322)**
- **Wan2.2 团队**
- **ComfyUI 社区**

---

## 许可证 / License

MIT License
