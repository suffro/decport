"""Qwen source backbone."""

from decport.backbones.base import HuggingFaceCausalBackbone


class QwenBackbone(HuggingFaceCausalBackbone):
    """Frozen Qwen3-0.6B representation extractor."""

    DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
