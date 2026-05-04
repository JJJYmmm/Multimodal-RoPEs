from .mrope import MRopeConfig


class OmniRopeConfig(MRopeConfig):
    """
    Configuration class for OmniRoPE.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        mrope_section: list[int] | None = None,
        temporal_stride: float = 1.0,
        **kwargs,
    ):
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        super().__init__(
            dim,
            base,
            mrope_section=mrope_section,
            temporal_stride=temporal_stride,
            **kwargs,
        )
        self.name = "omnirope"
