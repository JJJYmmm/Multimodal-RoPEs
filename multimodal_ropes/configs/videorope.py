from .mrope import MRopeConfig


class VideoRopeConfig(MRopeConfig):
    """
    Configuration class for VideoRoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] = [16, 24, 24],
        temporal_stride: int = 2,
        **kwargs,
    ):
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "videorope"
