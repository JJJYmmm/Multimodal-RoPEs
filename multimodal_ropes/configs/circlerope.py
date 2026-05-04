from .mrope import MRopeConfig


class CircleRopeConfig(MRopeConfig):
    """
    Configuration class for CircleRoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] | None = None,
        temporal_stride: float = 1.0,
        move_to_origin: bool = False,
        move_to_positive: bool = False,
        dff_rate: bool = False,
        method: str = "circle",
        radius: float = -1,
        alpha: float = -1,
        **kwargs,
    ):
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        if method not in {"circle", "no_circle"}:
            raise ValueError("method must be one of: 'circle', 'no_circle'.")
        if alpha != -1 and not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1], or -1 for the default.")
        if not isinstance(dff_rate, bool | int | float):
            raise TypeError("dff_rate must be a bool or numeric interpolation weight.")
        if not 0 <= float(dff_rate) <= 1:
            raise ValueError("dff_rate must be in [0, 1].")
        super().__init__(dim, base, mrope_section, temporal_stride, **kwargs)
        self.name = "circlerope"
        self.move_to_origin = move_to_origin
        self.move_to_positive = move_to_positive
        self.dff_rate = dff_rate
        self.method = method
        self.radius = radius
        self.alpha = alpha
