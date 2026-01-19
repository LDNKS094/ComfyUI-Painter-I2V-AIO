# modules/painteri2v_extend/__init__.py
# SOURCE TRACKING: Based on ComfyUI-PainterLongVideo, refactored for new API

from .nodes import PainterI2VExtend

NODE_CLASS_MAPPINGS = {
    "PainterI2VExtend": PainterI2VExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2VExtend": "PainterI2V Extend (Video Continuation)",
}
