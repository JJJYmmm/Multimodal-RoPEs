from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
