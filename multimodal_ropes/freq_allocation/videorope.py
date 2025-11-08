import torch

from ..configs.videorope import VideoRopeConfig
from .mrope import MRopeEmbedding


class VideoRopeEmbedding(MRopeEmbedding):
    def __init__(self, config, device=None, extra_config: VideoRopeConfig = None):
        super().__init__(config, device, extra_config)

    def apply_transformation(self, freqs, mrope_section):
        """Follow the order of hwhwhwhwtttt... to reorganize the frequency layout.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 2
            idx = slice(offset, length, 2)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
