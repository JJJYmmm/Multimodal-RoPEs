class VanillaRoPEConfig:
    """
    Configuration class for RoPE.
    """

    def __init__(self, dim: int, base: int = 10000):
        self.name = "vanilla-rope"
        self.dim = dim
        self.base = base

    def __repr__(self):
        return f"RoPEConfig(dim={self.dim}, base={self.base})"