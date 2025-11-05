import torch
from torch import nn

from collections.abc import Callable
from typing import Optional
from transformers.modeling_rope_utils import dynamic_rope_update, ROPE_INIT_FUNCTIONS

from ..configs.vanilla import VanillaRopeConfig


class RopeEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None, extra_config: VanillaRopeConfig = None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.extra_config = extra_config

        self.rope_type = self.config.rope_scaling["rope_type"]
        self.rope_init_fn: Callable = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 3:
            position_ids = position_ids[
                0
            ]  # vanilla rope does not have multiple position ids for one token
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
