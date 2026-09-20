"""SmolLM target backbone."""

from decport.backbones.base import HuggingFaceCausalBackbone


class SmolLMBackbone(HuggingFaceCausalBackbone):
    """Frozen SmolLM2-360M-Instruct representation extractor."""

    DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
