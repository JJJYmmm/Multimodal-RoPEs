from .mrope import MRopeConfig


class CircleRopeConfig(MRopeConfig):
    """
    Configuration class for CircleRoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] = [16, 24, 24],
        temporal_stride: float = 1.0,
        move_to_origin: bool = False,
        move_to_positive: bool = False,
        dff_rate: bool = False,
        method: str = "circle",
        radius: float = -1,
        alpha: float = -1,
        **kwargs,
    ):
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "circlerope"
        self.move_to_origin = move_to_origin
        self.move_to_positive = move_to_positive
        self.dff_rate = dff_rate
        self.method = method
        self.radius = radius
        self.alpha = alpha
