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
        temporal_stride: int = 1,
        num_key_value_heads: int = 8,
        **kwargs,
    ):
        """
        Configuration class for MHRoPE.
        mrope_section means the number of heads for each dimension, like [2, 3, 3] for T, H, W.
        """
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "mhrope"
        self.num_key_value_heads = num_key_value_heads
