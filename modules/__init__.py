# -*- coding: utf-8 -*-
"""
PainterAIO Modules

All node modules are organized here for easy management and upstream sync.
"""

from .wan22fmlf import (
    WanFirstMiddleLastFrameToVideo,
    WanMultiFrameRefToVideo,
    WanMultiImageLoader,
    WanFourFrameReferenceUltimate,
    WanAdvancedI2V,
    WanAdvancedExtractLastFrames,
    WanAdvancedExtractLastImages,
)

from .painteri2v import PainterI2V
from .painteri2v_advanced import PainterI2VAdvanced
from .paintersampler import PainterSampler
from .paintersampler_advanced import PainterSamplerAdvanced
from .painterflf2v import PainterFLF2V
from .painterlongvideo import PainterLongVideo

__all__ = [
    # wan22fmlf
    "WanFirstMiddleLastFrameToVideo",
    "WanMultiFrameRefToVideo",
    "WanMultiImageLoader",
    "WanFourFrameReferenceUltimate",
    "WanAdvancedI2V",
    "WanAdvancedExtractLastFrames",
    "WanAdvancedExtractLastImages",
    # painteri2v
    "PainterI2V",
    # painteri2v_advanced
    "PainterI2VAdvanced",
    # paintersampler
    "PainterSampler",
    # paintersampler_advanced
    "PainterSamplerAdvanced",
    # painterflf2v
    "PainterFLF2V",
    # painterlongvideo
    "PainterLongVideo",
]
