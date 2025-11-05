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
        spatial_reset=False,
        **kwargs,
    ):
        super().__init__(dim, base, mrope_section, **kwargs)
        self.name = "mrope-interleave"
        self.spatial_reset = spatial_reset

    def __repr__(self):
        return f"MRoPE_Interleave_Config(dim={self.dim}, base={self.base}, mrope_section={self.mrope_section}, spatial_reset={self.spatial_reset}, temporal_stride={self.temporal_stride})"
