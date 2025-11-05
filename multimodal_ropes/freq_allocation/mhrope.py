import torch

from ..configs.mhrope import MHRopeConfig
from .mrope import MRopeEmbedding


class MHRopeEmbedding(MRopeEmbedding):
    def __init__(self, config, device=None, extra_config: MHRopeConfig = None):
        super().__init__(config, device, extra_config)

    def apply_transformation(self, freqs, mrope_section):
        """Apply Multi-Head RoPE to 3D rotary embeddings.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """

        batch_size, seq_length, dim = freqs.shape[1:]
        freqs = torch.cat(
            [
                freqs[m, :, None, :, :].repeat(1, num, 1, 1)
                for m, num in enumerate(mrope_section)
            ],
            dim=1,
        )
        if (
            sum(mrope_section) < self.extra_config.num_key_value_heads
        ):  # padding unused heads with zeros, e.g. q * cos(embed) + rotate_half(q) * sin(embed) = q
            freqs = torch.cat(
                [
                    freqs,
                    torch.zeros(
                        batch_size,
                        self.extra_config.num_key_value_heads - sum(mrope_section),
                        seq_length,
                        dim,
                        device=freqs.device,
                        dtype=freqs.dtype,
                    ),
                ],
                dim=1,
            )

        return freqs
