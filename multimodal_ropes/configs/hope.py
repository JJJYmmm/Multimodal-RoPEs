from .mrope import MRopeConfig


class HopeConfig(MRopeConfig):
    """
    Configuration class for HoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] | None = None,
        temporal_stride: float = 2.0,
        temporal_stride_lst: list[float] | None = None,
        **kwargs,
    ):
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        if temporal_stride_lst is None:
            temporal_stride_lst = [0.5, 0.75, 1.0, 1.25, 1.5]
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "hope"
        self.temporal_stride_lst = temporal_stride_lst
