from .mrope_i import MRopeInterleaveConfig


class ILRopeConfig(MRopeInterleaveConfig):
    """
    Configuration class for IL-RoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] | None = None,
        temporal_stride: float = 1.0,
        spatial_reset: bool = True,
        **kwargs,
    ):
        if not spatial_reset:
            raise ValueError(
                "ILRoPE uses a reset visual layout in this package. Use "
                "'mrope-interleave' for the non-reset interleaved layout."
            )
        if mrope_section is None:
            mrope_section = [24, 20, 20]
        super().__init__(
            dim,
            base,
            mrope_section=mrope_section,
            temporal_stride=temporal_stride,
            spatial_reset=spatial_reset,
            **kwargs,
        )
        self.name = "ilrope"
