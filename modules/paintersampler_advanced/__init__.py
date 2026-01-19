# -*- coding: utf-8 -*-
"""
PainterSamplerAdvanced Node

SOURCE TRACKING:
  - Based on: ref/Comfyui-PainterSampler/painter_sampler.py
  - Last synced: N/A (new node)

MODIFICATIONS:
  - New node with 4 conditioning inputs for dual-phase sampling
  - Designed to work with PainterI2VAdvanced outputs
"""

from .painter_sampler_advanced import PainterSamplerAdvanced

__all__ = ["PainterSamplerAdvanced"]
