from functools import partial

from .configs import *
from .pos_design import *
from .freq_allocation import *

import logging

logging.basicConfig(level=logging.INFO)

SUPPORT_MM_ROPES = [
    "vanilla-rope",
    "mrope",
    "mrope-interleave",
    "mhrope",
]


MAPPINGS_NAME_TO_CONFIG = {
    "vanilla-rope": VanillaRopeConfig,
    "mrope": MRopeConfig,
    "mrope-interleave": MRopeInterleaveConfig,
    "mhrope": MHRopeConfig,
}


MAPPINGS_NAME_TO_POS_DESIGN = {
    "vanilla-rope": get_vanilla_rope_index,
    "mrope": get_mrope_index,
    "mrope-interleave": get_mrope_interleave_index,
    "mhrope": get_mrope_interleave_index,
}


MAPPINGS_NAME_TO_FREQ_ALLOCATION = {
    "vanilla-rope": RopeEmbedding,
    "mrope": MRopeEmbedding,
    "mrope-interleave": MRopeInterleaveEmbedding,
    "mhrope": MHRopeEmbedding,
}


def get_multimodal_rope_config(rope_name: str, **kwargs) -> VanillaRopeConfig:
    assert rope_name in SUPPORT_MM_ROPES, f"RoPE type {rope_name} not supported."
    rope_config_class = MAPPINGS_NAME_TO_CONFIG[rope_name]
    return rope_config_class(**kwargs)


def get_multimodal_rope(rope_name: str, *args, **kwargs):
    assert rope_name in SUPPORT_MM_ROPES, f"RoPE type {rope_name} not supported."

    rope_config_class = MAPPINGS_NAME_TO_CONFIG[rope_name]
    config = rope_config_class(*args, **kwargs)

    logging.info(f"Config: {config}")

    pos_design_func = MAPPINGS_NAME_TO_POS_DESIGN[config.name]
    freq_allocation_class = MAPPINGS_NAME_TO_FREQ_ALLOCATION[config.name]

    def patched_pos_design_func(*args, **kwargs):
        return pos_design_func(extra_config=config, *args, **kwargs)

    rope_embed_factory = partial(freq_allocation_class, extra_config=config)

    return patched_pos_design_func, rope_embed_factory
