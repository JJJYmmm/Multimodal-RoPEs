import torch

from ..configs.mrope_i import MRopeInterleaveConfig
from .mrope import MRopeEmbedding


class MRopeInterleaveEmbedding(MRopeEmbedding):
    def __init__(self, config, device=None, extra_config: MRopeInterleaveConfig = None):
        super().__init__(config, device, extra_config)

    def apply_transformation(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
