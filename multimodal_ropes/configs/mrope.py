from .vanilla import VanillaRoPEConfig

class MRoPEConfig(VanillaRoPEConfig):
    """
    Configuration class for MRoPE/MRoPE-I.
    """

    def __init__(self, dim: int, base: int = 10000,
                 mrope_section: list[int] = [24, 20, 20],
                 spatial_reset: bool = False):
        super().__init__(dim, base)
        self.mrope_section = mrope_section
        self.spatial_reset = spatial_reset
        if self.spatial_reset:
            self.name = "mrope-interleave"
        else:
            self.name = "mrope"

    def __repr__(self):
        return f"MRoPEConfig(dim={self.dim}, base={self.base}, mrope_section={self.mrope_section}, spatial_reset={self.spatial_reset})"