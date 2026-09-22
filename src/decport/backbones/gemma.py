"""Gemma target backbone."""

from decport.backbones.base import HuggingFaceCausalBackbone


class GemmaBackbone(HuggingFaceCausalBackbone):
    """Frozen Gemma 3 270M instruction representation extractor."""

    # The Google repository is license-gated. This is an ungated safetensors mirror
    # of the same instruction-tuned checkpoint and is pinned in experiment provenance.
    DEFAULT_MODEL_ID = "unsloth/gemma-3-270m-it"
