"""TSM (TimeStep Master) ComfyUI custom node.

Provides the TimeStepMasterLoader node for loading LyCORIS TSM checkpoints
with per-step asymmetric mixture of timestep LoRA experts.

Installation:
    Copy this directory to ComfyUI/custom_nodes/tsm_loader/
    Restart ComfyUI.
"""

from .tsm_loader import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
