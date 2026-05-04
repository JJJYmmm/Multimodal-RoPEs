import logging

from .configs.circlerope import CircleRopeConfig
from .configs.grape import GRAPEConfig
from .configs.hope import HopeConfig
from .configs.ilrope import ILRopeConfig
from .configs.mmrope import MMRopeConfig
from .configs.mrope import MRopeConfig
from .configs.mrope_i import MRopeInterleaveConfig
from .configs.mhrope import MHRopeConfig
from .configs.omnirope import OmniRopeConfig
from .configs.vanilla import VanillaRopeConfig
from .configs.v2pe import V2PEConfig
from .configs.videorope import VideoRopeConfig
from .freq_allocation.grape import GRAPEEmbedding
from .freq_allocation.hope import HopeEmbedding
from .freq_allocation.mmrope import MMRopeEmbedding
from .freq_allocation.mhrope import MHRopeEmbedding
from .freq_allocation.mrope import MRopeEmbedding
from .freq_allocation.mrope_i import MRopeInterleaveEmbedding
from .freq_allocation.vanilla import RopeEmbedding
from .freq_allocation.videorope import VideoRopeEmbedding
from .pos_design.circlerope import get_circlerope_index
from .pos_design.hope import get_hope_index
from .pos_design.ilrope import get_ilrope_index
from .pos_design.mmrope import get_mmrope_index
from .pos_design.mrope import get_mrope_index
from .pos_design.mrope_i import get_mrope_interleave_index
from .pos_design.omnirope import get_omnirope_index
from .pos_design.vanilla import get_vanilla_rope_index
from .pos_design.v2pe import get_v2pe_index
from .pos_design.videorope import get_videorope_index

SUPPORT_MM_ROPES = [
    "vanilla-rope",
    "mrope",
    "mrope-interleave",
    "mhrope",
    "videorope",
    "hope",
    "circlerope",
    "v2pe",
    "ilrope",
    "omnirope",
    "mmrope",
    "grape",
]


MAPPINGS_NAME_TO_CONFIG = {
    "vanilla-rope": VanillaRopeConfig,
    "mrope": MRopeConfig,
    "mrope-interleave": MRopeInterleaveConfig,
    "mhrope": MHRopeConfig,
    "videorope": VideoRopeConfig,
    "hope": HopeConfig,
    "circlerope": CircleRopeConfig,
    "v2pe": V2PEConfig,
    "ilrope": ILRopeConfig,
    "omnirope": OmniRopeConfig,
    "mmrope": MMRopeConfig,
    "grape": GRAPEConfig,
}


MAPPINGS_NAME_TO_POS_DESIGN = {
    "vanilla-rope": get_vanilla_rope_index,
    "mrope": get_mrope_index,
    "mrope-interleave": get_mrope_interleave_index,
    "mhrope": get_mrope_interleave_index,
    "videorope": get_videorope_index,
    "hope": get_hope_index,
    "circlerope": get_circlerope_index,
    "v2pe": get_v2pe_index,
    "ilrope": get_ilrope_index,
    "omnirope": get_omnirope_index,
    "mmrope": get_mmrope_index,
    "grape": get_mrope_index,
}


MAPPINGS_NAME_TO_FREQ_ALLOCATION = {
    "vanilla-rope": RopeEmbedding,
    "mrope": MRopeEmbedding,
    "mrope-interleave": MRopeInterleaveEmbedding,
    "mhrope": MHRopeEmbedding,
    "videorope": VideoRopeEmbedding,
    "hope": HopeEmbedding,
    "circlerope": MRopeEmbedding,
    "v2pe": RopeEmbedding,
    "ilrope": MRopeInterleaveEmbedding,
    "omnirope": MRopeEmbedding,
    "mmrope": MMRopeEmbedding,
    "grape": GRAPEEmbedding,
}


def get_multimodal_rope_config(rope_name: str, **kwargs) -> VanillaRopeConfig:
    assert rope_name in SUPPORT_MM_ROPES, f"RoPE type {rope_name} not supported."
    rope_config_class = MAPPINGS_NAME_TO_CONFIG[rope_name]
    return rope_config_class(**kwargs)


def get_multimodal_rope(rope_name: str, *args, **kwargs):
    assert rope_name in SUPPORT_MM_ROPES, f"RoPE type {rope_name} not supported."

    rope_config_class = MAPPINGS_NAME_TO_CONFIG[rope_name]
    config = rope_config_class(*args, **kwargs)

    logger = logging.getLogger(__name__)
    logger.debug("Config: %s", config)

    pos_design_func = MAPPINGS_NAME_TO_POS_DESIGN[config.name]
    freq_allocation_class = MAPPINGS_NAME_TO_FREQ_ALLOCATION[config.name]

    def patched_pos_design_func(*args, **kwargs):
        return pos_design_func(extra_config=config, *args, **kwargs)

    # Important: return a real class, not a functools.partial. Some downstream codepaths may
    # depend on type identity / class attributes.
    rope_config = config

    class RopeEmbeddingFactory(freq_allocation_class):
        def __init__(self, config, device=None, **inner_kwargs):
            super().__init__(
                config, device=device, extra_config=rope_config, **inner_kwargs
            )

    return patched_pos_design_func, RopeEmbeddingFactory
