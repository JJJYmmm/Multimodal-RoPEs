import torch
from torch import nn

from collections.abc import Callable

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from ..configs.vanilla import VanillaRopeConfig


class RopeEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self, config, device=None, extra_config: VanillaRopeConfig | None = None
    ):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.extra_config = extra_config

        # transformers==5.3.0: configs expose standardized `rope_parameters`.
        self.config.standardize_rope_params()
        rope_params = self.config.rope_parameters
        self.rope_type = rope_params.get("rope_type", "default")

        inv_freq, self.attention_scaling = self._compute_inv_freq(
            device=torch.device("cpu")
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def _compute_inv_freq(self, device: torch.device):
        rope_params = self.config.rope_parameters
        rope_type = rope_params.get("rope_type", "default")

        if rope_type == "default":
            base = float(rope_params.get("rope_theta", 10000.0))
            partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
            head_dim = (
                getattr(self.config, "head_dim", None)
                or self.config.hidden_size // self.config.num_attention_heads
            )
            dim = int(int(head_dim) * partial_rotary_factor)
            # Keep at least one rotary pair.
            dim = max(2, dim - (dim % 2))
            exponents = (
                torch.arange(0, dim, 2, device=device, dtype=torch.int64).to(
                    dtype=torch.float
                )
                / dim
            )
            base_t = torch.tensor(base, device=device, dtype=torch.float)
            inv_freq = torch.pow(base_t, exponents).reciprocal()
            attention_scaling = 1.0
            return inv_freq, attention_scaling

        # Other RoPE flavors via HF helpers.
        rope_init_fn: Callable = ROPE_INIT_FUNCTIONS[rope_type]
        return rope_init_fn(self.config, device)

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
