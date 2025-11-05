import torch

from transformers.modeling_rope_utils import dynamic_rope_update

from ..configs.mrope import MRopeConfig
from .vanilla import RopeEmbedding


class MRopeEmbedding(RopeEmbedding):
    def __init__(self, config, device=None, extra_config: MRopeConfig = None):
        super().__init__(config, device, extra_config)

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
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

    def apply_transformation(self, freqs, mrope_section):
        """Apply MRoPE to 3D rotary embeddings.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs = torch.cat(
            [m[i % 3] for i, m in enumerate(freqs.split(mrope_section, dim=-1))], dim=-1
        )
        return freqs
