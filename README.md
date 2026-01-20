[English](README.md) | [中文](readme_zh.md)

# ComfyUI-PainterAIO

All-in-One collection of Painter's ComfyUI nodes for Wan 2.2 video generation.

This project consolidates and refactors multiple ComfyUI video node projects with unified API and optimized code structure.

---

## Based On

This project is based on the following open-source projects:

| Project | Author | Feature |
|---------|--------|---------|
| [ComfyUI-PainterI2V](https://github.com/princepainter/ComfyUI-PainterI2V) | princepainter | I2V with motion enhancement |
| [ComfyUI-PainterI2Vadvanced](https://github.com/princepainter/ComfyUI-PainterI2Vadvanced) | princepainter | Dual sampler + color protection |
| [ComfyUI-PainterLongVideo](https://github.com/princepainter/ComfyUI-PainterLongVideo) | princepainter | Long video extension |
| [Comfyui-PainterFLF2V](https://github.com/princepainter/Comfyui-PainterFLF2V) | princepainter | First-last frame interpolation |
| [ComfyUI-Wan22FMLF](https://github.com/wallen0322/ComfyUI-Wan22FMLF) | wallen0322 | Multi-frame reference conditioning |

---

## Nodes

| Node | Description |
|------|-------------|
| **PainterI2V** | Image-to-video with I2V / FLF2V modes, motion enhancement |
| **PainterI2VAdvanced** | Advanced version with dual sampler output, color protection |
| **PainterI2VExtend** | Video extension for long video generation |
| **PainterSampler** | Sampler with motion enhancement |
| **PainterSamplerAdvanced** | Advanced sampler with dual sampler output |

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-PainterAIO.git
```

---

## Features

- **Motion Enhancement**: Fix slow-motion issues with 4-step LoRAs (e.g., lightx2v)
- **Color Protection**: Prevent green/dark color drift after motion enhancement
- **First-Last Frame Interpolation**: FLF2V mode with start and end frame anchoring
- **Long Video Extension**: Seamless continuation from previous video segment
- **SVI LoRA Compatible**: Supports latents_mean padding for SVI mode
- **Dual CLIP Vision**: Semantic guidance for both start and end frames

---

## Parameters

### motion_amplitude

| Motion Type | Recommended Value |
|-------------|-------------------|
| Subtle motion (talking, blinking) | 1.0 - 1.1 |
| Moderate motion (walking, gestures) | 1.1 - 1.2 |
| Large motion (running, jumping) | 1.2 - 1.5 |

### color_protect

Keep enabled (default: True) to prevent color drift after motion enhancement.

---

## Acknowledgements

- **[princepainter](https://github.com/princepainter)**
- **[wallen0322](https://github.com/wallen0322)**
- **Wan2.2 Team**
- **ComfyUI Community**

---

## License

MIT License
