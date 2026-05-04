from .mrope import MRopeConfig


class MMRopeConfig(MRopeConfig):
    """
    Configuration class for MM-RoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] | None = None,
        temporal_stride: float = 1.0,
        position_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs,
    ):
        if len(position_scale) != 3:
            raise ValueError("position_scale must contain exactly three values.")
        if any(scale <= 0 for scale in position_scale):
            raise ValueError("position_scale values must be positive.")
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        super().__init__(
            dim,
            base,
            mrope_section=mrope_section,
            temporal_stride=temporal_stride,
            **kwargs,
        )
        self.name = "mmrope"
        self.position_scale = position_scale
