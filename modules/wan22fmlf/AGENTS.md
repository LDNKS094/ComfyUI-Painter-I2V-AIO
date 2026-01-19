# WAN22FMLF MODULE

Multi-frame reference conditioning nodes for Wan2.2 A14B I2V models.

## NODES (7 total)

| Node | File | Lines | Purpose |
|------|------|-------|---------|
| WanAdvancedI2V | wan_advanced_i2v.py | 681 | Ultimate unified node with SVI/latent continue |
| WanAdvancedExtractLastFrames | wan_advanced_i2v.py | - | Extract last N latent frames |
| WanAdvancedExtractLastImages | wan_advanced_i2v.py | - | Extract last N image frames |
| WanFirstMiddleLastFrameToVideo | wan_first_middle_last.py | 405 | 3-frame reference (start/mid/end) |
| WanMultiFrameRefToVideo | wan_multi_frame.py | 296 | N-frame universal reference |
| WanFourFrameReferenceUltimate | wan_4_frame_ultimate.py | 284 | 4-frame with adjustable placeholder |
| WanMultiImageLoader | wan_multi_image_loader.py | 96 | UI multi-image batch loader |

## KEY PATTERNS

### Mask System
- `mask_high_noise` — controls high-noise stage denoising
- `mask_low_noise` — controls low-noise refinement stage
- Value 0.0 = full reference, 1.0 = full generation

### Structural Repulsion Boost
Spatial gradient conditioning for motion enhancement:
```python
if structural_repulsion_boost > 1.001:
    # Creates motion-aware mask from image differences
    spatial_gradient = create_spatial_gradient(img1, img2)
    mask_high_noise *= spatial_gradient
```

### Long Video Modes (WanAdvancedI2V)
| Mode | Behavior |
|------|----------|
| DISABLED | Standard single-clip generation |
| AUTO_CONTINUE | Use motion_frames for clip chaining |
| SVI | Latent-space conditioning (SVI Pro style) |
| LATENT_CONTINUE | Direct latent injection from prev_latent |

## WHERE TO LOOK

| Task | File |
|------|------|
| Add new frame conditioning | wan_first_middle_last.py (template) |
| Modify mask logic | All files have similar mask_high/low pattern |
| SVI mode changes | wan_advanced_i2v.py lines 176-380 |
| Frontend multi-loader | wan_multi_image_loader.py + js/ |

## UPSTREAM

Source: `ref/ComfyUI-Wan22FMLF/`
