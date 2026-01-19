# -*- coding: utf-8 -*-
"""
ComfyUI-PainterAIO
All-in-One collection of Painter's ComfyUI nodes for video generation

This is a fork/merge of multiple upstream projects:
  - ComfyUI-PainterI2V
  - ComfyUI-PainterI2Vadvanced
  - Comfyui-PainterSampler
  - Comfyui-PainterFLF2V (merged into PainterI2V)
  - ComfyUI-PainterLongVideo (merged into PainterI2VExtend)

Original sources are in ref/ directory for comparison during upstream sync.
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Import all node classes from modules
from .modules.painteri2v import PainterI2V
from .modules.painteri2v_extend import PainterI2VExtend
from .modules.painteri2v_advanced import PainterI2VAdvanced

# Traditional API nodes
from .modules.paintersampler import PainterSampler
from .modules.paintersampler_advanced import PainterSamplerAdvanced

__version__ = "0.1.0"

WEB_DIRECTORY = "./js"


class PainterAIOExtension(ComfyExtension):
    """Unified extension for all Painter AIO nodes"""

    @override
    async def get_node_list(self):
        # All nodes using new ComfyExtension API
        return [
            # painteri2v (T2V/I2V/FLF2V unified)
            PainterI2V,
            # painteri2v_extend (video continuation)
            PainterI2VExtend,
            # painteri2v_advanced (full control, 4 cond output)
            PainterI2VAdvanced,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterSampler": "Painter Sampler",
    "PainterSamplerAdvanced": "Painter Sampler Advanced",
}

__all__ = [
    "WEB_DIRECTORY",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
