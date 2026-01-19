# -*- coding: utf-8 -*-
"""
ComfyUI-PainterAIO
All-in-One collection of Painter's ComfyUI nodes for video generation

This is a fork/merge of multiple upstream projects:
  - ComfyUI-Wan22FMLF
  - ComfyUI-PainterI2V
  - ComfyUI-PainterI2Vadvanced
  - Comfyui-PainterSampler
  - Comfyui-PainterFLF2V
  - ComfyUI-PainterLongVideo

Original sources are in ref/ directory for comparison during upstream sync.
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Import all node classes from modules
from .modules.wan22fmlf import (
    WanFirstMiddleLastFrameToVideo,
    WanMultiFrameRefToVideo,
    WanMultiImageLoader,
    WanFourFrameReferenceUltimate,
    WanAdvancedI2V,
    WanAdvancedExtractLastFrames,
    WanAdvancedExtractLastImages,
)
from .modules.painteri2v import PainterI2V
from .modules.painteri2v_extend import PainterI2VExtend
from .modules.painteri2v_advanced import PainterI2VAdvanced
# PainterFLF2V functionality merged into PainterI2V (auto-detects FLF2V mode when end_image is provided)

# Traditional API nodes
from .modules.paintersampler import PainterSampler
from .modules.paintersampler_advanced import PainterSamplerAdvanced
from .modules.painterlongvideo import PainterLongVideo

__version__ = "0.1.0"

WEB_DIRECTORY = "./js"


class PainterAIOExtension(ComfyExtension):
    """Unified extension for all Painter AIO nodes"""

    @override
    async def get_node_list(self):
        # All nodes using new ComfyExtension API
        return [
            # wan22fmlf
            WanFirstMiddleLastFrameToVideo,
            WanMultiFrameRefToVideo,
            WanMultiImageLoader,
            WanFourFrameReferenceUltimate,
            WanAdvancedI2V,
            WanAdvancedExtractLastFrames,
            WanAdvancedExtractLastImages,
            # painteri2v
            PainterI2V,
            # painteri2v_extend
            PainterI2VExtend,
            # painteri2v_advanced
            PainterI2VAdvanced,
            # Note: PainterFLF2V merged into PainterI2V
        ]


async def comfy_entrypoint():
    """Entry point for ComfyUI new-style extension loading"""
    return PainterAIOExtension()


# ============================================================================
# Traditional API Support (for nodes not yet converted)
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # PainterSampler uses traditional API
    "PainterSampler": PainterSampler,
    "PainterSamplerAdvanced": PainterSamplerAdvanced,
    # PainterLongVideo uses traditional API
    "PainterLongVideo": PainterLongVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterSampler": "Painter Sampler",
    "PainterSamplerAdvanced": "Painter Sampler Advanced",
    "PainterLongVideo": "PainterLongVideo",
}

__all__ = [
    "WEB_DIRECTORY",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
