from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from multimodal_ropes.entry import get_multimodal_rope


GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "qwen35_rope_current.pt"


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
            return torch.stack([position_temporal, position_height, position_width], dim=0)

    return Stub()


def _build_inputs(device: torch.device):
    # Fixed multimodal layout:
    # text(5) + image(4) + text(3) + video(4) + text(2)
    text1, img, text2, vid, text3 = 5, 4, 3, 4, 2
    seq_len = text1 + img + text2 + vid + text3
    input_ids = (torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1) + 1000)

    mm = [0] * text1 + [1] * img + [0] * text2 + [2] * vid + [0] * text3
    mm_token_type_ids = torch.tensor([mm], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # With spatial_merge_size=2 and grid_thw=(1,4,4) => llm grid (1,2,2) => 4 vision tokens.
    image_grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    video_grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    return input_ids, mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw


def _compute_current():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Model,
        Qwen3_5TextRotaryEmbedding,
    )

    device = torch.device("cpu")
    stub = _stub_qwen35(spatial_merge_size=2)
    input_ids, mm_token_type_ids, attention_mask, image_grid_thw, video_grid_thw = _build_inputs(device)

    # Reference: HF get_rope_index (Qwen3.5 uses the same mm_token_type_ids grouping logic as Qwen3-VL).
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


def test_qwen35_golden_current():
    assert GOLDEN_PATH.exists(), f"Missing golden file: {GOLDEN_PATH}"
    golden = torch.load(GOLDEN_PATH, map_location="cpu")
    cur = _compute_current()

    torch.testing.assert_close(cur["position_ids"], golden["position_ids"], rtol=0, atol=0)
    torch.testing.assert_close(cur["rope_deltas"], golden["rope_deltas"], rtol=0, atol=0)

    torch.testing.assert_close(cur["hf_cos"], golden["hf_cos"], rtol=0, atol=0)
    torch.testing.assert_close(cur["hf_sin"], golden["hf_sin"], rtol=0, atol=0)
    torch.testing.assert_close(cur["ours_cos"], golden["ours_cos"], rtol=0, atol=0)
    torch.testing.assert_close(cur["ours_sin"], golden["ours_sin"], rtol=0, atol=0)

