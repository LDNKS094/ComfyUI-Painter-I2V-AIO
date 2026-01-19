# -*- coding: utf-8 -*-
"""
Wan Video Reference Nodes
Multi-frame reference conditioning for Wan2.2 A14B I2V models

SOURCE TRACKING:
  - upstream: ref/ComfyUI-Wan22FMLF/
  - Last synced: Initial copy
"""

from .wan_first_middle_last import WanFirstMiddleLastFrameToVideo
from .wan_multi_frame import WanMultiFrameRefToVideo
from .wan_multi_image_loader import WanMultiImageLoader
from .wan_4_frame_ultimate import WanFourFrameReferenceUltimate
from .wan_advanced_i2v import (
    WanAdvancedI2V,
    WanAdvancedExtractLastFrames,
    WanAdvancedExtractLastImages,
)

__all__ = [
    "WanFirstMiddleLastFrameToVideo",
    "WanMultiFrameRefToVideo",
    "WanMultiImageLoader",
    "WanFourFrameReferenceUltimate",
    "WanAdvancedI2V",
    "WanAdvancedExtractLastFrames",
    "WanAdvancedExtractLastImages",
]
