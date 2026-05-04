from .mrope import MRopeConfig


class VideoRopeConfig(MRopeConfig):
    """
    Configuration class for VideoRoPE.
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
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "videorope"
