from __future__ import annotations

import torch

from multimodal_ropes.entry import get_multimodal_rope


def test_qwen35_partial_rope_matches_hf_embedding():
    """
    Qwen3.5 uses partial RoPE (e.g. partial_rotary_factor=0.25) with interleaved MRoPE.
    This test ensures our patched rotary embedding matches HF's reference implementation
    for the same config.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    cfg = Qwen3_5TextConfig()
    cfg.rope_scaling = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
    }
    cfg.standardize_rope_params()

    hf_emb = Qwen3_5TextRotaryEmbedding(cfg)

    _, ours_cls = get_multimodal_rope(
        "mrope-interleave",
        dim=cfg.head_dim,
        base=10_000_000,
        mrope_section=[11, 11, 10],
        temporal_stride=1.0,
        spatial_reset=False,
    )
    ours_emb = ours_cls(cfg)

    x = torch.zeros((1, 1, cfg.head_dim), dtype=torch.float32)
    position_ids = torch.arange(0, 32, dtype=torch.long).view(1, -1)

    hf_cos, hf_sin = hf_emb(x, position_ids)
    ours_cos, ours_sin = ours_emb(x, position_ids)

    torch.testing.assert_close(ours_cos, hf_cos, rtol=0, atol=0)
    torch.testing.assert_close(ours_sin, hf_sin, rtol=0, atol=0)

