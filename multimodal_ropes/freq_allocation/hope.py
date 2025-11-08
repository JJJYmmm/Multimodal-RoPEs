import torch
from transformers.modeling_rope_utils import dynamic_rope_update

from ..configs.hope import HopeConfig
from .videorope import VideoRopeEmbedding


class HopeEmbedding(VideoRopeEmbedding):
    def __init__(self, config, device=None, extra_config: HopeConfig = None):
        super().__init__(config, device, extra_config)

    # HoPE follow the same frequency allocation as VideoRoPE, but apply NoPE to t dimension.
    # DIfferent from the official implementation, We achieve NoPE by setting the t pos to 0.
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # HoPE: set t pos to 0
        position_ids[0].masked_fill_(position_ids[0] > 0, 0)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            freqs = self.apply_transformation(
                freqs, self.extra_config.mrope_section
            )  # use extra_config to avoid confusion with self.mrope_section
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
