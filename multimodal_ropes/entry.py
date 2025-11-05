from functools import partial
from .configs import *
from . import pos_design
from . import freq_allocation

SUPPORT_MM_ROPES = [
    "vanilla-rope",
    "mrope",
]


MAPPINGS_NAME_TO_CONFIG = {
    "vanilla-rope": VanillaRoPEConfig,
}


MAPPINGS_NAME_TO_POS_DESIGN = {
    "vanilla-rope": pos_design.vanilla.get_rope_index,
    # "mrope": pos_design.mrope.get_rope_index,
}


MAPPINGS_NAME_TO_FREQ_ALLOCATION = {
    "vanilla-rope": freq_allocation.vanilla.RopeEmbedding,
    # "mrope": freq_allocation.mrope.RopeEmbedding,
}


def get_multimodal_rope_config(rope_name: str, **kwargs) -> VanillaRoPEConfig:
    assert rope_name in SUPPORT_MM_ROPES, f"RoPE type {rope_name} not supported."
    rope_config_class = MAPPINGS_NAME_TO_CONFIG[rope_name]
    return rope_config_class(**kwargs)


def get_multimodal_rope(rope_name: str, **kwargs):
    assert rope_name in SUPPORT_MM_ROPES, f"RoPE type {rope_name} not supported."

    rope_config_class = MAPPINGS_NAME_TO_CONFIG[rope_name]
    config = rope_config_class(**kwargs)

    pos_design_func = MAPPINGS_NAME_TO_POS_DESIGN[config.name]
    freq_allocation_class = MAPPINGS_NAME_TO_FREQ_ALLOCATION[config.name]

    rope_index = partial(pos_design_func, extra_config=config)
    rope_embed_factory = partial(freq_allocation_class, config=config)

    return rope_index, rope_embed_factory