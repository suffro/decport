"""TinyLlama target backbone."""

from decport.backbones.base import HuggingFaceCausalBackbone


class LlamaBackbone(HuggingFaceCausalBackbone):
    """Frozen Llama-family causal LM used as the third heterogeneous target."""

    DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
