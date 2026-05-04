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
        if len(mrope_section) != 3:
            raise ValueError("mrope_section must contain exactly three values.")
        if any(section <= 0 for section in mrope_section):
            raise ValueError("mrope_section values must be positive.")
        if temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive.")
        super().__init__(dim, base, **kwargs)
        self.name = "mrope"
        self.mrope_section = mrope_section
        self.temporal_stride = temporal_stride
