from .mrope import MRopeConfig


class MRopeInterleaveConfig(MRopeConfig):
    """
    Configuration class for MRoPE/MRoPE-I.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] = [24, 20, 20],
        temporal_stride: float = 1.0,
        spatial_reset: bool = False,
        **kwargs,
    ):
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "mrope-interleave"
        self.spatial_reset = spatial_reset
