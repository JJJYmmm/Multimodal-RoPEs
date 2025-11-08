from .vanilla import VanillaRopeConfig


class MRopeConfig(VanillaRopeConfig):
    """
    Configuration class for MRoPE/MRoPE-I.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] = [16, 24, 24],
        temporal_stride: float = 2.0,
        **kwargs,
    ):
        super().__init__(dim, base, **kwargs)
        self.name = "mrope"
        self.mrope_section = mrope_section
        self.temporal_stride = temporal_stride
