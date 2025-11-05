from .vanilla import VanillaRopeConfig


class MRopeConfig(VanillaRopeConfig):
    """
    Configuration class for MRoPE/MRoPE-I.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] = [24, 20, 20],
        temporal_stride: int = 2,
        **kwargs,
    ):
        super().__init__(dim, base, **kwargs)
        self.name = "mrope"
        self.mrope_section = mrope_section
        self.temporal_stride = temporal_stride

    def __repr__(self):
        return f"MRoPEConfig(dim={self.dim}, base={self.base}, mrope_section={self.mrope_section}, temporal_stride={self.temporal_stride})"
