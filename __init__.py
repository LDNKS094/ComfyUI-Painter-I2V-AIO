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

# Import all node classes from modules (all V3 API)
from .modules.painteri2v import PainterI2V
from .modules.painteri2v_extend import PainterI2VExtend
from .modules.painteri2v_advanced import PainterI2VAdvanced
from .modules.paintersampler import PainterSampler
from .modules.paintersampler_advanced import PainterSamplerAdvanced


class PainterAIOExtension(ComfyExtension):
    """Unified extension for all Painter AIO nodes"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PainterI2V,
            PainterI2VExtend,
            PainterI2VAdvanced,
            PainterSampler,
            PainterSamplerAdvanced,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    """Entry point for ComfyUI new-style extension loading"""
    return PainterAIOExtension()

