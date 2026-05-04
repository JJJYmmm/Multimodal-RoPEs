from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from multimodal_ropes.entry import SUPPORT_MM_ROPES, get_multimodal_rope


GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "qwen3vl_rope_hf4571.pt"


def _build_canonical_inputs(device: torch.device):
    cfg = Qwen3VLConfig()
    merge = cfg.vision_config.spatial_merge_size

    vision_start = int(cfg.vision_start_token_id)
    vision_end = int(cfg.vision_end_token_id)
    image_tok = int(cfg.image_token_id)
    video_tok = int(cfg.video_token_id)

    # grid_thw=(1,4,4), merge=2 -> llm grid (1,2,2) => 4 vision tokens.
    image_grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    video_grid_thw = torch.tensor([[2, 4, 4]], device=device, dtype=torch.long)
    vision_len = 1 * (4 // merge) * (4 // merge)

    toks: list[int] = []
    toks += [100, 101, 102, 103, 104]
    toks += [vision_start]
    toks += [image_tok] * vision_len
    toks += [vision_end]
    toks += [200, 201]
    toks += [300, vision_start]
    toks += [video_tok] * vision_len
    toks += [vision_end]
    toks += [400]
    toks += [301, vision_start]
    toks += [video_tok] * vision_len
    toks += [vision_end]
    toks += [500]

    input_ids = torch.tensor([toks], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    mm = []
    for t in toks:
        if t == image_tok:
            mm.append(1)
        elif t == video_tok:
            mm.append(2)
        else:
            mm.append(0)
    mm_token_type_ids = torch.tensor([mm], device=device, dtype=torch.long)

    return {
        "qwen3vl_config": cfg,
        "input_ids": input_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "attention_mask": attention_mask,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }


def _compute_current():
    device = torch.device("cpu")
    inp = _build_canonical_inputs(device)

    stub = SimpleNamespace(config=inp["qwen3vl_config"])

    common_kwargs = dict(dim=128, base=5_000_000)
    per_rope_kwargs = {
        "vanilla-rope": dict(common_kwargs),
        "mrope": dict(common_kwargs, mrope_section=[16, 24, 24], temporal_stride=2),
        "mrope-interleave": dict(
            common_kwargs,
            mrope_section=[24, 20, 20],
            temporal_stride=1,
            spatial_reset=False,
        ),
        "mhrope": dict(
            common_kwargs,
            num_key_value_heads=8,
            mrope_section=[2, 3, 3],
            temporal_stride=1,
            spatial_reset=True,
        ),
        "videorope": dict(
            common_kwargs, mrope_section=[16, 24, 24], temporal_stride=2.0
        ),
        "hope": dict(
            common_kwargs,
            mrope_section=[16, 24, 24],
            temporal_stride=2.0,
            temporal_stride_lst=[0.5, 0.75, 1.0, 1.25, 1.5],
        ),
        "circlerope": dict(
            common_kwargs,
            mrope_section=[16, 24, 24],
            temporal_stride=1.0,
            move_to_origin=True,
            dff_rate=True,
            method="circle",
            radius=10,
            alpha=0.5,
        ),
        "v2pe": dict(common_kwargs, visual_stride=16.0),
        "ilrope": dict(
            common_kwargs,
            mrope_section=[24, 20, 20],
            temporal_stride=1.0,
            spatial_reset=True,
        ),
        "omnirope": dict(
            common_kwargs,
            mrope_section=[16, 24, 24],
            temporal_stride=1.0,
        ),
        "mmrope": dict(
            common_kwargs,
            mrope_section=[16, 24, 24],
            temporal_stride=1.0,
            position_scale=(4.0, 8.0, 8.0),
        ),
        "grape": dict(common_kwargs),
    }

    txt_cfg = Qwen3VLTextConfig()
    # Match the baseline (HF 4.57.1) config knobs used to generate the golden.
    if getattr(txt_cfg, "rope_scaling", None) is None:
        txt_cfg.rope_scaling = {}
    txt_cfg.rope_scaling["rope_type"] = "default"
    txt_cfg.rope_scaling["rope_theta"] = 5_000_000.0
    x = torch.zeros((1, 1, txt_cfg.head_dim), dtype=torch.float32)

    out = {}
    for rope_name in SUPPORT_MM_ROPES:
        rope_index, rope_embed_cls = get_multimodal_rope(
            rope_name, **per_rope_kwargs[rope_name]
        )
        pos_ids, deltas = rope_index(
            stub,
            input_ids=inp["input_ids"],
            mm_token_type_ids=inp["mm_token_type_ids"],
            image_grid_thw=inp["image_grid_thw"],
            video_grid_thw=inp["video_grid_thw"],
            attention_mask=inp["attention_mask"],
        )

        emb = rope_embed_cls(txt_cfg)
        cos, sin = emb(x, pos_ids)

        out[rope_name] = {
            "position_ids": pos_ids.cpu(),
            "rope_deltas": deltas.cpu(),
            "cos": cos.cpu(),
            "sin": sin.cpu(),
        }
    return out


def _stub_qwen35(spatial_merge_size: int = 2):
    class Stub:
        def __init__(self):
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size)
            )

        def get_vision_position_ids(
            self,
            start_position: int,
            grid_thw: torch.Tensor,
            temp_merge_size: int = 1,
            spatial_merge_size: int = 1,
            time_interval: int = 1,
            device: str | torch.device | None = None,
        ):
            llm_grid_t, llm_grid_h, llm_grid_w = (
                grid_thw[0].item() // temp_merge_size,
                grid_thw[1].item() // spatial_merge_size,
                grid_thw[2].item() // spatial_merge_size,
            )
            image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
            position_width = torch.arange(
                start_position, start_position + llm_grid_w, device=device
            ).repeat(llm_grid_h * llm_grid_t)
            position_height = torch.arange(
                start_position, start_position + llm_grid_h, device=device
            ).repeat_interleave(llm_grid_w * llm_grid_t)
            position_temporal = torch.full(
                (image_seq_length,),
                start_position,
                device=device,
                dtype=torch.long,
            )
            position_temporal = position_temporal * time_interval
            return torch.stack(
                [position_temporal, position_height, position_width], dim=0
            )

    return Stub()


def _build_qwen35_inputs(device: torch.device):
    text1, img, text2, vid, text3 = 5, 4, 3, 4, 2
    seq_len = text1 + img + text2 + vid + text3
    input_ids = (
        torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1) + 1000
    )

    mm = [0] * text1 + [1] * img + [0] * text2 + [2] * vid + [0] * text3
    mm_token_type_ids = torch.tensor([mm], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    image_grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    video_grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    return input_ids, mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw


def _compute_qwen35_current():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Model,
        Qwen3_5TextRotaryEmbedding,
    )

    device = torch.device("cpu")
    stub = _stub_qwen35(spatial_merge_size=2)
    input_ids, mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw = (
        _build_qwen35_inputs(device)
    )

    pos_ids, deltas = Qwen3_5Model.get_rope_index(
        stub,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )

    cfg = Qwen3_5TextConfig()
    cfg.rope_scaling = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
    }
    cfg.standardize_rope_params()

    x = torch.zeros((1, 1, cfg.head_dim), dtype=torch.float32)

    hf_emb = Qwen3_5TextRotaryEmbedding(cfg)
    hf_cos, hf_sin = hf_emb(x, pos_ids)

    _, ours_cls = get_multimodal_rope(
        "mrope-interleave",
        dim=cfg.head_dim,
        base=10_000_000,
        mrope_section=[11, 11, 10],
        temporal_stride=1.0,
        spatial_reset=False,
    )
    ours_emb = ours_cls(cfg)
    ours_cos, ours_sin = ours_emb(x, pos_ids)

    return {
        "position_ids": pos_ids.cpu(),
        "rope_deltas": deltas.cpu(),
        "hf_cos": hf_cos.cpu(),
        "hf_sin": hf_sin.cpu(),
        "ours_cos": ours_cos.cpu(),
        "ours_sin": ours_sin.cpu(),
    }


def test_rope_matches_hf4571_golden():
    assert GOLDEN_PATH.exists(), (
        f"Missing golden file: {GOLDEN_PATH}. "
        "Generate it with transformers==4.57.1 on the pre-upgrade repo state, then commit it."
    )

    golden = torch.load(GOLDEN_PATH, map_location="cpu")
    expected = golden["expected"]
    current = _compute_current()

    for rope_name in SUPPORT_MM_ROPES:
        exp = expected[rope_name]
        cur = current[rope_name]

        torch.testing.assert_close(
            cur["position_ids"], exp["position_ids"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            cur["rope_deltas"], exp["rope_deltas"], rtol=0, atol=0
        )

        # Rotary is float; allow tiny drift across PyTorch builds.
        torch.testing.assert_close(cur["cos"], exp["cos"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(cur["sin"], exp["sin"], rtol=1e-5, atol=1e-5)


def test_qwen35_matches_unified_golden():
    assert GOLDEN_PATH.exists(), f"Missing golden file: {GOLDEN_PATH}"
    golden = torch.load(GOLDEN_PATH, map_location="cpu")
    exp = golden["qwen35_expected"]
    cur = _compute_qwen35_current()

    torch.testing.assert_close(cur["position_ids"], exp["position_ids"], rtol=0, atol=0)
    torch.testing.assert_close(cur["rope_deltas"], exp["rope_deltas"], rtol=0, atol=0)
    torch.testing.assert_close(cur["hf_cos"], exp["hf_cos"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cur["hf_sin"], exp["hf_sin"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cur["ours_cos"], exp["ours_cos"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cur["ours_sin"], exp["ours_sin"], rtol=1e-5, atol=1e-5)


def _positions(name: str, **kwargs):
    inp = _build_canonical_inputs(torch.device("cpu"))
    stub = SimpleNamespace(config=inp["qwen3vl_config"])
    rope_index, _ = get_multimodal_rope(name, dim=128, base=5_000_000, **kwargs)
    pos_ids, deltas = rope_index(
        stub,
        input_ids=inp["input_ids"],
        mm_token_type_ids=inp["mm_token_type_ids"],
        image_grid_thw=inp["image_grid_thw"],
        video_grid_thw=inp["video_grid_thw"],
        attention_mask=inp["attention_mask"],
    )
    return pos_ids[:, 0], deltas[0, 0]


def test_new_variant_position_rules():
    v2pe_pos_ids, delta = _positions("v2pe", visual_stride=1.0)
    torch.testing.assert_close(
        v2pe_pos_ids[:, 6:10],
        torch.tensor(
            [
                [5.25, 5.50, 5.75, 6.00],
                [5.25, 5.50, 5.75, 6.00],
                [5.25, 5.50, 5.75, 6.00],
            ]
        ),
    )
    assert delta == -9

    il_pos_ids, _ = _positions(
        "ilrope", mrope_section=[24, 20, 20], temporal_stride=1.0
    )
    omni_pos_ids, _ = _positions(
        "omnirope", mrope_section=[16, 24, 24], temporal_stride=1.0
    )
    torch.testing.assert_close(il_pos_ids, omni_pos_ids)
    torch.testing.assert_close(
        il_pos_ids[:, 6:10],
        torch.tensor([[6, 6, 6, 6], [0, 0, 1, 1], [0, 1, 0, 1]]).float(),
    )

    mm_pos_ids, _ = _positions(
        "mmrope",
        mrope_section=[16, 24, 24],
        temporal_stride=1.0,
        position_scale=(4.0, 8.0, 8.0),
    )
    torch.testing.assert_close(
        mm_pos_ids[:, 6:10],
        torch.tensor([[6, 6, 6, 6], [6, 6, 14, 14], [6, 14, 6, 14]]).float(),
    )

    grape_pos_ids, _ = _positions(
        "grape", mrope_section=[16, 24, 24], temporal_stride=1.0
    )
    torch.testing.assert_close(
        grape_pos_ids[:, 6:10],
        torch.tensor([[6, 6, 6, 6], [6, 6, 7, 7], [6, 7, 6, 7]]),
    )


def test_omnirope_temporal_stride_advances_visual_block_extent():
    cfg = Qwen3VLConfig()
    stub = SimpleNamespace(config=cfg)
    merge = cfg.vision_config.spatial_merge_size
    image_grid_thw = torch.tensor([[3, 4, 4]], dtype=torch.long)
    visual_len = 3 * (4 // merge) * (4 // merge)
    input_ids = torch.tensor(
        [[cfg.vision_start_token_id] + [cfg.image_token_id] * visual_len + [1234]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor([[0] + [1] * visual_len + [0]], dtype=torch.long)

    rope_index, _ = get_multimodal_rope(
        "omnirope",
        dim=128,
        base=5_000_000,
        temporal_stride=2.0,
    )
    pos_ids, _ = rope_index(
        stub,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )

    torch.testing.assert_close(pos_ids[:, 0, -1], torch.tensor([6.0, 6.0, 6.0]))


def test_new_variant_config_validation():
    with pytest.raises(ValueError, match="not supported"):
        get_multimodal_rope("unknown-rope", dim=128)

    with pytest.raises(ValueError, match="mrope_section"):
        get_multimodal_rope("mrope", dim=128, mrope_section=[16, 24])

    with pytest.raises(ValueError, match="temporal_stride"):
        get_multimodal_rope("videorope", dim=128, temporal_stride=0)

    with pytest.raises(ValueError, match="num_key_value_heads"):
        get_multimodal_rope(
            "mhrope",
            dim=128,
            mrope_section=[3, 3, 3],
            num_key_value_heads=8,
        )

    with pytest.raises(ValueError, match="temporal_stride_lst"):
        get_multimodal_rope("hope", dim=128, temporal_stride_lst=[])

    with pytest.raises(ValueError, match="method"):
        get_multimodal_rope("circlerope", dim=128, method="spiral")

    with pytest.raises(ValueError, match="visual_stride"):
        get_multimodal_rope("v2pe", dim=128, visual_stride=0)

    with pytest.raises(ValueError, match="reset visual layout"):
        get_multimodal_rope("ilrope", dim=128, spatial_reset=False)

    with pytest.raises(ValueError, match="position_scale"):
        get_multimodal_rope("mmrope", dim=128, position_scale=(1.0, 2.0))

    with pytest.raises(ValueError, match="position_scale values"):
        get_multimodal_rope("mmrope", dim=128, position_scale=(1.0, -1.0, 1.0))


def test_interleaved_frequency_allocations_do_not_mutate_inputs():
    cfg = Qwen3VLTextConfig()
    cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5_000_000.0}
    freqs = torch.stack(
        [
            torch.full((1, 1, 8), 10.0),
            torch.full((1, 1, 8), 20.0),
            torch.full((1, 1, 8), 30.0),
        ]
    )

    for rope_name in ("mrope-interleave", "videorope"):
        _, rope_cls = get_multimodal_rope(
            rope_name,
            dim=128,
            base=5_000_000,
            mrope_section=[2, 2, 2],
        )
        emb = rope_cls(cfg)
        original = freqs.clone()
        emb.apply_transformation(freqs, [2, 2, 2])
        torch.testing.assert_close(freqs, original)


def test_mmrope_frequency_pattern_matches_meta_component():
    _, rope_cls = get_multimodal_rope(
        "mmrope",
        dim=128,
        base=5_000_000,
        mrope_section=[16, 24, 24],
        temporal_stride=1.0,
    )
    cfg = Qwen3VLTextConfig()
    cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5_000_000.0}
    emb = rope_cls(cfg)
    freqs = torch.stack(
        [
            torch.full((1, 1, 8), 10.0),
            torch.full((1, 1, 8), 20.0),
            torch.full((1, 1, 8), 30.0),
        ]
    )

    out = emb.apply_transformation(freqs, [16, 24, 24])

    torch.testing.assert_close(
        out[0, 0],
        torch.tensor([10, 10, 20, 30, 20, 30, 20, 30]).float(),
    )


def test_grape_frequency_allocation_variants():
    freqs = torch.stack(
        [
            torch.full((1, 1, 8), 10.0),
            torch.full((1, 1, 8), 20.0),
            torch.full((1, 1, 8), 30.0),
        ]
    )
    cfg = Qwen3VLTextConfig()
    cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5_000_000.0}

    _, interleaved_cls = get_multimodal_rope(
        "grape",
        dim=128,
        base=5_000_000,
        frequency_allocation="interleaved",
    )
    interleaved = interleaved_cls(cfg)
    torch.testing.assert_close(
        interleaved.apply_transformation(freqs, [16, 24, 24])[0, 0],
        torch.tensor([10, 20, 30, 10, 20, 30, 10, 20]).float(),
    )


def test_grape_learned_plane_modes_apply_qk():
    pos_ids, _ = _positions("grape", mrope_section=[16, 24, 24], temporal_stride=1.0)
    pos_ids = pos_ids.unsqueeze(1)
    seq_len = pos_ids.shape[-1]

    cfg = Qwen3VLTextConfig()
    cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5_000_000.0}
    q = torch.randn(1, 2, seq_len, cfg.head_dim)
    k = torch.randn_like(q)

    for mode in ("axis", "mixed", "block_mixed"):
        _, rope_cls = get_multimodal_rope(
            "grape",
            dim=128,
            base=5_000_000,
            grape_mode=mode,
            block_size=16,
        )
        emb = rope_cls(cfg)
        q_out, k_out = emb.apply_qk(q, k, pos_ids)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert q_out.dtype == q.dtype
        assert k_out.dtype == k.dtype

        with pytest.raises(RuntimeError, match="rotate q/k directly"):
            emb(q, pos_ids)


def test_grape_learned_plane_modes_keep_zero_position_identity():
    cfg = Qwen3VLTextConfig()
    cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5_000_000.0}
    q = torch.randn(1, 2, 7, cfg.head_dim)
    k = torch.randn_like(q)
    zero_pos = torch.zeros(3, 1, 7)

    for mode in ("axis", "mixed", "block_mixed"):
        _, rope_cls = get_multimodal_rope(
            "grape",
            dim=128,
            base=5_000_000,
            grape_mode=mode,
            block_size=16,
        )
        emb = rope_cls(cfg)
        q_out, k_out = emb.apply_qk(q, k, zero_pos)

        torch.testing.assert_close(q_out, q)
        torch.testing.assert_close(k_out, k)


def test_grape_learned_modes_use_text_config_frequencies():
    cfg = Qwen3VLTextConfig()
    cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5_000_000.0}

    _, rope_cls = get_multimodal_rope(
        "grape",
        dim=128,
        base=10_000,
        grape_mode="mixed",
    )
    emb = rope_cls(cfg)

    torch.testing.assert_close(
        emb.learned_rotation.inv_freq,
        emb.inv_freq,
    )

    _, block_cls = get_multimodal_rope(
        "grape",
        dim=128,
        base=10_000,
        grape_mode="block_mixed",
        block_size=16,
    )
    block_emb = block_cls(cfg)
    torch.testing.assert_close(
        block_emb.learned_rotation.inv_freq,
        block_emb.inv_freq.reshape(8, 8),
    )


def test_grape_learned_modes_reject_2d_positions():
    cfg = Qwen3VLTextConfig()
    cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5_000_000.0}
    q = torch.randn(1, 2, 7, cfg.head_dim)
    k = torch.randn_like(q)
    position_ids = torch.arange(7).view(1, -1)

    _, rope_cls = get_multimodal_rope(
        "grape",
        dim=128,
        base=5_000_000,
        grape_mode="block_mixed",
    )
    emb = rope_cls(cfg)

    with pytest.raises(ValueError, match="explicit 3D position_ids"):
        emb.apply_qk(q, k, position_ids)
