from .vanilla import VanillaRopeConfig


class MRopeConfig(VanillaRopeConfig):
    """
    Configuration class for MRoPE/MRoPE-I.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] | None = None,
        temporal_stride: float = 2.0,
        **kwargs,
    ):
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        super().__init__(dim, base, **kwargs)
        self.name = "mrope"
        self.mrope_section = mrope_section
        self.temporal_stride = temporal_stride
