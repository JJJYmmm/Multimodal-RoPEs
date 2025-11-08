from .mrope import MRopeConfig


class HopeConfig(MRopeConfig):
    """
    Configuration class for HoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] = [16, 24, 24],
        temporal_stride: int = 2,
        temporal_stride_lst: list[int] = [0.5, 0.75, 1.0, 1.25, 1.5],
        **kwargs,
    ):
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "hope"
        self.temporal_stride_lst = temporal_stride_lst
