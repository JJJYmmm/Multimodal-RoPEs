from .mrope import MRopeConfig


class GRAPEConfig(MRopeConfig):
    """
    Configuration class for multimodal GRAPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] | None = None,
        temporal_stride: float = 1.0,
        log_freq_scale: float = 1.0,
        frequency_allocation: str = "chunked",
        grape_mode: str = "canonical",
        num_planes: int | None = None,
        block_size: int = 16,
        planes_per_block: int | None = None,
        rope_init: bool = True,
        learnable: bool = True,
        **kwargs,
    ):
        if log_freq_scale <= 0:
            raise ValueError("log_freq_scale must be positive.")
        if frequency_allocation not in {"chunked", "interleaved"}:
            raise ValueError(
                "frequency_allocation must be one of: 'chunked', 'interleaved'."
            )
        if grape_mode not in {"canonical", "axis", "mixed", "block_mixed"}:
            raise ValueError(
                "grape_mode must be one of: 'canonical', 'axis', 'mixed', "
                "'block_mixed'."
            )
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        if num_planes is not None and num_planes <= 0:
            raise ValueError("num_planes must be positive when provided.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if planes_per_block is not None and planes_per_block <= 0:
            raise ValueError("planes_per_block must be positive when provided.")
        super().__init__(
            dim,
            base,
            mrope_section=mrope_section,
            temporal_stride=temporal_stride,
            **kwargs,
        )
        self.name = "grape"
        self.log_freq_scale = log_freq_scale
        self.frequency_allocation = frequency_allocation
        self.grape_mode = grape_mode
        self.num_planes = num_planes
        self.block_size = block_size
        self.planes_per_block = planes_per_block
        self.rope_init = rope_init
        self.learnable = learnable
