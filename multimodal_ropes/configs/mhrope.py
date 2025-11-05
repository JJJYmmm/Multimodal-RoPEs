from .mrope_i import MRopeInterleaveConfig


class MHRopeConfig(MRopeInterleaveConfig):
    """
    Configuration class for MRoPE/MRoPE-I.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] = [2, 3, 3],
        num_key_value_heads: int = 8,
        **kwargs,
    ):
        super().__init__(dim, base, mrope_section, **kwargs)
        self.name = "mhrope"
        self.num_key_value_heads = num_key_value_heads

    def __repr__(self):
        return f"MHRopeConfig(dim={self.dim}, base={self.base}, mrope_section={self.mrope_section}, spatial_reset={self.spatial_reset}, num_key_value_heads={self.num_key_value_heads}, temporal_stride={self.temporal_stride})"
