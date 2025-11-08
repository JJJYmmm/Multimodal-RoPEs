class VanillaRopeConfig:
    """
    Configuration class for RoPE.
    """

    def __init__(self, dim: int, base: int = 10000, **kwargs):
        self.name = "vanilla-rope"
        self.dim = dim
        self.base = base

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"
