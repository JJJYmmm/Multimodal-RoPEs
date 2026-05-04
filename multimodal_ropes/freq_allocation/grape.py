import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_rope_utils import dynamic_rope_update

from ..configs.grape import GRAPEConfig
from .mrope import MRopeEmbedding


def _make_inv_freq(
    num_planes: int,
    base: float | None,
    log_freq_scale: float,
    source_inv_freq: torch.Tensor | None = None,
):
    if num_planes <= 0:
        raise ValueError("num_planes must be positive.")
    if source_inv_freq is not None:
        inv_freq = source_inv_freq.detach().float()
        if inv_freq.numel() != num_planes:
            source_index = torch.linspace(0, inv_freq.numel() - 1, num_planes)
            inv_freq = inv_freq[source_index.round().long()]
    else:
        if base is None:
            raise ValueError("base is required when source_inv_freq is not provided.")
        exponents = torch.arange(num_planes, dtype=torch.float) / num_planes
        inv_freq = torch.pow(
            torch.tensor(base, dtype=torch.float), exponents
        ).reciprocal()
    if log_freq_scale != 1.0:
        inv_freq = torch.exp(inv_freq.log() / log_freq_scale)
    return inv_freq


def _make_block_inv_freq(
    num_blocks: int,
    planes_per_block: int,
    base: float | None,
    log_freq_scale: float,
    source_inv_freq: torch.Tensor | None = None,
):
    if source_inv_freq is None:
        inv_freq = _make_inv_freq(planes_per_block, base, log_freq_scale)
        return inv_freq.view(1, -1).expand(num_blocks, -1).contiguous()

    source_inv_freq = source_inv_freq.detach().float()
    if source_inv_freq.numel() % num_blocks != 0:
        inv_freq = _make_inv_freq(
            num_blocks * planes_per_block,
            base,
            log_freq_scale,
            source_inv_freq,
        )
        return inv_freq.reshape(num_blocks, planes_per_block)

    source_blocks = source_inv_freq.reshape(num_blocks, -1)
    if source_blocks.shape[1] == planes_per_block:
        inv_freq = source_blocks
    else:
        source_index = torch.linspace(0, source_blocks.shape[1] - 1, planes_per_block)
        inv_freq = source_blocks[:, source_index.round().long()]

    if log_freq_scale != 1.0:
        inv_freq = torch.exp(inv_freq.log() / log_freq_scale)
    return inv_freq


def _init_dense_planes(a: torch.Tensor, b: torch.Tensor):
    a.zero_()
    b.zero_()
    dim = a.shape[-1]
    for plane in range(a.shape[-2]):
        pair = (2 * plane) % dim
        a[..., plane, pair] = 1.0
        b[..., plane, (pair + 1) % dim] = 1.0


def _init_axis_weight(axis_weight: torch.Tensor):
    axis_weight.zero_()
    flat = axis_weight.reshape(-1, 3)
    for plane in range(flat.shape[0]):
        flat[plane, plane % 3] = 1.0


class MultiPlaneGrapeRotation(nn.Module):
    """Learned GRAPE rank-2 planes for one coordinate axis."""

    def __init__(
        self,
        head_dim: int,
        num_planes: int,
        base: float | None,
        log_freq_scale: float,
        rope_init: bool,
        learnable: bool,
        source_inv_freq: torch.Tensor | None = None,
    ):
        super().__init__()
        if num_planes > head_dim // 2:
            raise ValueError("num_planes cannot exceed head_dim // 2.")
        self.head_dim = head_dim
        self.num_planes = num_planes
        self.a = nn.Parameter(torch.empty(num_planes, head_dim))
        self.b = nn.Parameter(torch.empty(num_planes, head_dim))
        self.a.requires_grad_(learnable)
        self.b.requires_grad_(learnable)
        self.register_buffer(
            "inv_freq",
            _make_inv_freq(num_planes, base, log_freq_scale, source_inv_freq),
            persistent=False,
        )
        self.reset_parameters(rope_init)

    def reset_parameters(self, rope_init: bool):
        with torch.no_grad():
            if rope_init:
                _init_dense_planes(self.a, self.b)
            else:
                nn.init.normal_(self.a, std=0.02)
                nn.init.normal_(self.b, std=0.02)

    def get_orthonormal_planes(self):
        a = F.normalize(self.a, dim=-1)
        b = self.b - (self.b * a).sum(dim=-1, keepdim=True) * a
        b = F.normalize(b, dim=-1)
        return a, b

    def forward(self, x, pos):
        a, b = self.get_orthonormal_planes()
        a = a.to(device=x.device, dtype=x.dtype)
        b = b.to(device=x.device, dtype=x.dtype)
        xa = torch.einsum("blhd,pd->blhp", x, a)
        xb = torch.einsum("blhd,pd->blhp", x, b)
        angle = (
            pos[..., None].to(x.device, dtype=torch.float)
            * self.inv_freq.to(x.device)[None, None, :]
        )
        cos = angle.cos().unsqueeze(2)
        sin = angle.sin().unsqueeze(2)
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        xa_rot = xa * cos - xb * sin
        xb_rot = xa * sin + xb * cos
        delta = torch.einsum("blhp,pd->blhd", xa_rot - xa, a) + torch.einsum(
            "blhp,pd->blhd", xb_rot - xb, b
        )
        return x + delta


class AxisGrapeRotation(nn.Module):
    """GRAPE-MRoPE: one learned-plane bank per t/h/w axis."""

    def __init__(
        self,
        head_dim: int,
        num_planes: int,
        base: float | None,
        log_freq_scale: float,
        rope_init: bool,
        learnable: bool,
        source_inv_freq: torch.Tensor | None = None,
    ):
        super().__init__()
        self.t_grape = MultiPlaneGrapeRotation(
            head_dim,
            num_planes,
            base,
            log_freq_scale,
            rope_init,
            learnable,
            source_inv_freq,
        )
        self.h_grape = MultiPlaneGrapeRotation(
            head_dim,
            num_planes,
            base,
            log_freq_scale,
            rope_init,
            learnable,
            source_inv_freq,
        )
        self.w_grape = MultiPlaneGrapeRotation(
            head_dim,
            num_planes,
            base,
            log_freq_scale,
            rope_init,
            learnable,
            source_inv_freq,
        )

    def forward(self, x, positions):
        x = self.t_grape(x, positions[..., 0])
        x = self.h_grape(x, positions[..., 1])
        x = self.w_grape(x, positions[..., 2])
        return x


class MixedAxisGrapeRotation(nn.Module):
    """Learned planes with learned mixed t/h/w phase directions."""

    def __init__(
        self,
        head_dim: int,
        num_planes: int,
        base: float | None,
        log_freq_scale: float,
        rope_init: bool,
        learnable: bool,
        source_inv_freq: torch.Tensor | None = None,
    ):
        super().__init__()
        if num_planes > head_dim // 2:
            raise ValueError("num_planes cannot exceed head_dim // 2.")
        self.head_dim = head_dim
        self.num_planes = num_planes
        self.a = nn.Parameter(torch.empty(num_planes, head_dim))
        self.b = nn.Parameter(torch.empty(num_planes, head_dim))
        self.axis_weight = nn.Parameter(torch.empty(num_planes, 3))
        self.a.requires_grad_(learnable)
        self.b.requires_grad_(learnable)
        self.axis_weight.requires_grad_(learnable)
        self.register_buffer(
            "inv_freq",
            _make_inv_freq(num_planes, base, log_freq_scale, source_inv_freq),
            persistent=False,
        )
        self.reset_parameters(rope_init)

    def reset_parameters(self, rope_init: bool):
        with torch.no_grad():
            if rope_init:
                _init_dense_planes(self.a, self.b)
                _init_axis_weight(self.axis_weight)
            else:
                nn.init.normal_(self.a, std=0.02)
                nn.init.normal_(self.b, std=0.02)
                nn.init.normal_(self.axis_weight, std=0.02)

    def get_orthonormal_planes(self):
        a = F.normalize(self.a, dim=-1)
        b = self.b - (self.b * a).sum(dim=-1, keepdim=True) * a
        b = F.normalize(b, dim=-1)
        return a, b

    def forward(self, x, positions):
        a, b = self.get_orthonormal_planes()
        a = a.to(device=x.device, dtype=x.dtype)
        b = b.to(device=x.device, dtype=x.dtype)
        xa = torch.einsum("blhd,pd->blhp", x, a)
        xb = torch.einsum("blhd,pd->blhp", x, b)
        axis_weight = F.normalize(self.axis_weight, dim=-1)
        mixed_pos = torch.einsum(
            "blc,pc->blp", positions.to(x.device, dtype=torch.float), axis_weight
        )
        angle = mixed_pos * self.inv_freq.to(x.device)[None, None, :]
        cos = angle.cos().unsqueeze(2)
        sin = angle.sin().unsqueeze(2)
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        xa_rot = xa * cos - xb * sin
        xb_rot = xa * sin + xb * cos
        delta = torch.einsum("blhp,pd->blhd", xa_rot - xa, a) + torch.einsum(
            "blhp,pd->blhd", xb_rot - xb, b
        )
        return x + delta


class BlockMixedGrapeRotation(nn.Module):
    """Engineering-friendly learned GRAPE with local block planes."""

    def __init__(
        self,
        head_dim: int,
        block_size: int,
        planes_per_block: int,
        base: float | None,
        log_freq_scale: float,
        rope_init: bool,
        learnable: bool,
        source_inv_freq: torch.Tensor | None = None,
    ):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if head_dim % block_size != 0:
            raise ValueError("head_dim must be divisible by block_size.")
        if planes_per_block <= 0:
            raise ValueError("planes_per_block must be positive.")
        if planes_per_block > block_size // 2:
            raise ValueError("planes_per_block cannot exceed block_size // 2.")
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = head_dim // block_size
        self.planes_per_block = planes_per_block
        self.a = nn.Parameter(
            torch.empty(self.num_blocks, planes_per_block, block_size)
        )
        self.b = nn.Parameter(
            torch.empty(self.num_blocks, planes_per_block, block_size)
        )
        self.axis_weight = nn.Parameter(
            torch.empty(self.num_blocks, planes_per_block, 3)
        )
        self.a.requires_grad_(learnable)
        self.b.requires_grad_(learnable)
        self.axis_weight.requires_grad_(learnable)
        self.register_buffer(
            "inv_freq",
            _make_block_inv_freq(
                self.num_blocks,
                planes_per_block,
                base,
                log_freq_scale,
                source_inv_freq,
            ),
            persistent=False,
        )
        self.reset_parameters(rope_init)

    def reset_parameters(self, rope_init: bool):
        with torch.no_grad():
            if rope_init:
                _init_dense_planes(self.a, self.b)
                _init_axis_weight(self.axis_weight)
            else:
                nn.init.normal_(self.a, std=0.02)
                nn.init.normal_(self.b, std=0.02)
                nn.init.normal_(self.axis_weight, std=0.02)

    def get_orthonormal_planes(self):
        a = F.normalize(self.a, dim=-1)
        b = self.b - (self.b * a).sum(dim=-1, keepdim=True) * a
        b = F.normalize(b, dim=-1)
        return a, b

    def forward(self, x, positions):
        batch, seq_len, num_heads, head_dim = x.shape
        x_block = x.reshape(batch, seq_len, num_heads, self.num_blocks, self.block_size)
        a, b = self.get_orthonormal_planes()
        a = a.to(device=x.device, dtype=x.dtype)
        b = b.to(device=x.device, dtype=x.dtype)
        xa = torch.einsum("blhnd,npd->blhnp", x_block, a)
        xb = torch.einsum("blhnd,npd->blhnp", x_block, b)
        axis_weight = F.normalize(self.axis_weight, dim=-1)
        mixed_pos = torch.einsum(
            "blc,npc->blnp", positions.to(x.device, dtype=torch.float), axis_weight
        )
        angle = (
            mixed_pos.unsqueeze(2) * self.inv_freq.to(x.device)[None, None, None, :, :]
        )
        cos = angle.cos()
        sin = angle.sin()
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        xa_rot = xa * cos - xb * sin
        xb_rot = xa * sin + xb * cos
        delta = torch.einsum("blhnp,npd->blhnd", xa_rot - xa, a) + torch.einsum(
            "blhnp,npd->blhnd", xb_rot - xb, b
        )
        return (x_block + delta).reshape(batch, seq_len, num_heads, head_dim)


class GRAPEEmbedding(MRopeEmbedding):
    def __init__(self, config, device=None, extra_config: GRAPEConfig = None):
        super().__init__(config, device, extra_config)
        self.grape_mode = extra_config.grape_mode
        self.rotary_dim = int(self.inv_freq.numel() * 2)
        self.learned_rotation = None
        learned_inv_freq = self.inv_freq

        if self.grape_mode == "axis":
            self.learned_rotation = AxisGrapeRotation(
                self.rotary_dim,
                extra_config.num_planes or self.rotary_dim // 2,
                None,
                extra_config.log_freq_scale,
                extra_config.rope_init,
                extra_config.learnable,
                learned_inv_freq,
            )
        elif self.grape_mode == "mixed":
            self.learned_rotation = MixedAxisGrapeRotation(
                self.rotary_dim,
                extra_config.num_planes or self.rotary_dim // 2,
                None,
                extra_config.log_freq_scale,
                extra_config.rope_init,
                extra_config.learnable,
                learned_inv_freq,
            )
        elif self.grape_mode == "block_mixed":
            block_size = extra_config.block_size
            planes_per_block = extra_config.planes_per_block or block_size // 2
            self.learned_rotation = BlockMixedGrapeRotation(
                self.rotary_dim,
                block_size,
                planes_per_block,
                None,
                extra_config.log_freq_scale,
                extra_config.rope_init,
                extra_config.learnable,
                learned_inv_freq,
            )

    def _select_axis_by_pattern(self, freqs, pattern):
        pattern = torch.tensor(pattern, device=freqs.device, dtype=torch.long)
        axis_ids = pattern[
            torch.arange(freqs.shape[-1], device=freqs.device) % len(pattern)
        ]
        gather_index = axis_ids.view(1, 1, -1, 1).expand(
            freqs.shape[1], freqs.shape[2], -1, 1
        )
        return freqs.permute(1, 2, 3, 0).gather(-1, gather_index).squeeze(-1)

    def apply_transformation(self, freqs, mrope_section):
        """Apply a fixed commuting multimodal GRAPE plane assignment.

        Appendix G of the GRAPE paper defines 3D GRAPE as a product of three
        commuting generators. In this drop-in cos/sin implementation those
        generators are canonical 2D planes selected by `frequency_allocation`.
        """
        if self.extra_config.frequency_allocation == "chunked":
            return super().apply_transformation(freqs, mrope_section)
        if self.extra_config.frequency_allocation == "interleaved":
            return self._select_axis_by_pattern(freqs, (0, 1, 2))
        raise ValueError(
            f"Unknown GRAPE frequency allocation: {self.extra_config.frequency_allocation}"
        )

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if self.grape_mode != "canonical":
            raise RuntimeError(
                "Learned GRAPE modes rotate q/k directly and cannot be represented "
                "as a standalone cos/sin tensor. Use GRAPEEmbedding.apply_qk(q, k, "
                "position_ids) or patch the attention module to call it."
            )
        inv_freq = self.inv_freq.to(x.device)
        if self.extra_config.log_freq_scale != 1.0:
            inv_freq = torch.exp(inv_freq.log() / self.extra_config.log_freq_scale)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            if position_ids.ndim == 3:
                inv_freq_expanded = (
                    inv_freq[None, None, :, None]
                    .float()
                    .expand(3, position_ids.shape[1], -1, 1)
                )
                position_ids_expanded = position_ids[:, :, None, :].to(
                    device=x.device, dtype=torch.float
                )
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(2, 3)
                freqs = self.apply_transformation(
                    freqs, self.extra_config.mrope_section
                )
            else:
                inv_freq_expanded = (
                    inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
                )
                position_ids_expanded = position_ids[:, None, :].to(
                    device=x.device, dtype=torch.float
                )
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)

            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _prepare_positions(self, position_ids, x):
        if position_ids.ndim == 2:
            raise ValueError(
                "Learned GRAPE modes require explicit 3D position_ids shaped "
                "[3, batch, seq_len]; use canonical mode for 1D/2D positions."
            )
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                "Learned GRAPE expects position_ids shaped [3, batch, seq_len] "
                "for MRoPE-style (t, h, w) coordinates."
            )
        return position_ids.permute(1, 2, 0).to(device=x.device, dtype=torch.float)

    def _apply_to_tensor(self, x, position_ids):
        if self.learned_rotation is None:
            raise RuntimeError("apply_qk is only available for learned GRAPE modes.")
        if x.shape[-1] < self.rotary_dim:
            raise ValueError(
                f"Input head_dim {x.shape[-1]} is smaller than rotary_dim "
                f"{self.rotary_dim}."
            )
        positions = self._prepare_positions(position_ids, x)
        x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        x_rot = x_rot.transpose(1, 2)
        x_rot = self.learned_rotation(x_rot, positions).transpose(1, 2)
        if x_pass.shape[-1] == 0:
            return x_rot
        return torch.cat((x_rot, x_pass), dim=-1)

    def apply_qk(self, q, k, position_ids):
        """Apply learned multimodal GRAPE directly to query/key tensors.

        Args:
            q, k: attention tensors with shape [batch, heads, seq_len, head_dim].
            position_ids: MRoPE-style coordinates with shape [3, batch, seq_len].
        """
        return self._apply_to_tensor(q, position_ids), self._apply_to_tensor(
            k, position_ids
        )
