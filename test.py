import argparse
import logging
import re
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextRotaryEmbedding as HFQwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)
from transformers.utils import logging as hf_logging

from multimodal_ropes.entry import SUPPORT_MM_ROPES, get_multimodal_rope


logger = logging.getLogger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multihead_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
):
    # GQA: repeat freqs from KV heads to match query heads
    n_repeat = q.shape[1] // cos.shape[1]
    q_embed = (q * cos.repeat_interleave(n_repeat, dim=1)) + (
        rotate_half(q) * sin.repeat_interleave(n_repeat, dim=1)
    )
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def default_ckpt_dir() -> Path:
    # Keep checkpoints location stable: sibling `checkpoints/` next to the repo folder.
    repo_root = Path(__file__).resolve().parent
    return repo_root.parent / "checkpoints" / "Qwen3-VL-2B-Instruct"


def make_demo_image(size: int = 448) -> Image.Image:
    # Deterministic colorful image (no network dependency).
    img = Image.new("RGB", (size, size), (245, 238, 230))
    draw = ImageDraw.Draw(img)
    pad = size // 14
    cell = (size - 2 * pad) // 3
    colors = [
        (238, 99, 82),
        (83, 181, 226),
        (253, 202, 64),
        (145, 97, 226),
        (106, 196, 140),
        (244, 162, 97),
        (40, 157, 143),
        (231, 111, 81),
        (233, 196, 106),
    ]
    idx = 0
    for r in range(3):
        for c in range(3):
            x0 = pad + c * cell + cell // 10
            y0 = pad + r * cell + cell // 10
            x1 = x0 + cell - cell // 5
            y1 = y0 + cell - cell // 5
            draw.rounded_rectangle(
                [x0, y0, x1, y1], radius=cell // 6, fill=colors[idx % len(colors)]
            )
            # frosting-like swirl
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            for k in range(4):
                rr = cell // 4 - k * (cell // 18)
                draw.ellipse(
                    [cx - rr, cy - rr, cx + rr, cy + rr],
                    outline=(255, 255, 255),
                    width=3,
                )
            idx += 1
    return img


def pick_device_settings() -> dict[str, Any]:
    has_flash_attn = False
    try:
        import flash_attn  # noqa: F401

        has_flash_attn = True
    except Exception:
        has_flash_attn = False

    if has_flash_attn:
        return dict(
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    return dict(
        attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto"
    )


def patch_rope(model: Qwen3VLForConditionalGeneration, rope_name: str, **rope_kwargs):
    rope_index, rope_embed_cls = get_multimodal_rope(rope_name, **rope_kwargs)

    # Position-id design is used during generation via `model.model.get_rope_index(...)`.
    Qwen3VLModel.get_rope_index = rope_index

    # Cache the exact modules we want to patch (only Qwen3-VL text attention rotary modules).
    if not hasattr(model, "_mm_rope_rotary_modules"):
        targets = []
        for module in model.modules():
            rotary = getattr(module, "rotary_emb", None)
            if rotary is not None and isinstance(rotary, HFQwen3VLTextRotaryEmbedding):
                targets.append(module)
        setattr(model, "_mm_rope_rotary_modules", targets)

    # Swap all per-layer text rotary embeddings in-place (no need to re-load the model weights).
    replaced = 0
    for module in getattr(model, "_mm_rope_rotary_modules"):
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            continue

        try:
            cfg = rotary.config
        except Exception:
            continue

        new_rotary = rope_embed_cls(cfg)
        # Keep the device aligned with the parent attention module.
        try:
            dev = next(module.parameters()).device
            new_rotary = new_rotary.to(dev)
        except StopIteration:
            pass

        module.rotary_emb = new_rotary
        replaced += 1

    # Clear rope_deltas cache between variants.
    try:
        model.model.rope_deltas = None
    except Exception:
        pass

    return replaced


def run_caption(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = (
        generated_ids[:, prompt_len:]
        if generated_ids.shape[1] > prompt_len
        else generated_ids
    )
    text = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, default=default_ckpt_dir())
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-variant patch details and loading logs.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw decoded text (may include newlines).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    # Keep stdout clean: transformers 5.3 prints a detailed LOAD REPORT at WARNING level.
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    ckpt_dir = args.ckpt_dir
    if not (ckpt_dir / "config.json").exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_dir}")

    settings = pick_device_settings()
    logger.info("Loading model from %s (%s)", ckpt_dir, settings)
    model = Qwen3VLForConditionalGeneration.from_pretrained(ckpt_dir, **settings)
    processor = AutoProcessor.from_pretrained(ckpt_dir)

    img = make_demo_image()
    common_kwargs = dict(dim=128, base=5_000_000)

    results: dict[str, str] = {}
    for rope_name in SUPPORT_MM_ROPES:
        # Default per-variant kwargs (kept close to the existing repo defaults).
        kwargs: dict[str, Any] = dict(common_kwargs)
        if rope_name in {"mrope", "videorope", "hope"}:
            kwargs.update(mrope_section=[16, 24, 24], temporal_stride=2.0)
        if rope_name == "mrope-interleave":
            kwargs.update(
                mrope_section=[24, 20, 20], temporal_stride=1, spatial_reset=False
            )
        if rope_name == "mhrope":
            kwargs.update(
                num_key_value_heads=8,
                mrope_section=[2, 3, 3],
                temporal_stride=1,
                spatial_reset=True,
            )

        # Patch RoPE.
        replaced = patch_rope(model, rope_name, **kwargs)

        # MHRoPE needs a different apply_rotary implementation.
        restore_apply = None
        if rope_name == "mhrope":
            transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb = (
                apply_multihead_rotary_pos_emb
            )
            restore_apply = hf_apply_rotary_pos_emb

        try:
            caption = run_caption(
                model, processor, img, max_new_tokens=args.max_new_tokens
            )
        finally:
            if restore_apply is not None:
                transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb = (
                    restore_apply
                )

        results[rope_name] = caption
        if args.verbose:
            logger.info("[%s] replaced_rotary=%s", rope_name, replaced)

    for rope_name in SUPPORT_MM_ROPES:
        text = results[rope_name]
        if not args.raw:
            text = re.sub(r"\\s+", " ", text).strip()
        print(f"{rope_name}: {text}")


if __name__ == "__main__":
    main()
