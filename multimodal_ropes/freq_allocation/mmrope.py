import torch

from ..configs.mmrope import MMRopeConfig
from .mrope import MRopeEmbedding


class MMRopeEmbedding(MRopeEmbedding):
    def __init__(self, config, device=None, extra_config: MMRopeConfig = None):
        super().__init__(config, device, extra_config)

    def apply_transformation(self, freqs, mrope_section):
        """Apply MM-RoPE's distributed 2:3:3 meta-component allocation.

        Each meta component contains eight rotary pairs (16 channels):
        [T, T, H, W, H, W, H, W]. Remainder pairs keep the same pattern.
        """
        pattern = torch.tensor(
            (0, 0, 1, 2, 1, 2, 1, 2), device=freqs.device, dtype=torch.long
        )
        axis_ids = pattern[torch.arange(freqs.shape[-1], device=freqs.device) % 8]
        gather_index = axis_ids.view(1, 1, -1, 1).expand(
            freqs.shape[1], freqs.shape[2], -1, 1
        )
        return freqs.permute(1, 2, 3, 0).gather(-1, gather_index).squeeze(-1)
