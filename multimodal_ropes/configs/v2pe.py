from .vanilla import VanillaRopeConfig


class V2PEConfig(VanillaRopeConfig):
    """
    Configuration class for V2PE.

    visual_stride is the total RoPE-position span assigned to one visual block.
    The per-token visual step is visual_stride / num_visual_tokens, matching the
    official V2PE implementation's rope_pos_id_stride / num_image_token rule.
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        visual_stride: float = 16.0,
        **kwargs,
    ):
        if visual_stride <= 0:
            raise ValueError("visual_stride must be positive.")
        super().__init__(dim, base, **kwargs)
        self.name = "v2pe"
        self.visual_stride = visual_stride
