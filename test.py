import torch
import transformers
from transformers import AutoProcessor, Qwen3VLModel, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb as original_apply_rotary_pos_emb,
)
from multimodal_ropes.entry import get_multimodal_rope

import logging

logging.basicConfig(level=logging.INFO)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multihead_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # repeat interleave for GQA, (num_heads / num_key_value_heads) querys -> 1 key, repeat freqs of key
    # q [bs, num_heads, seq_len, head_dim]
    # k [bs, num_key_value_heads, seq_len, head_dim]
    # cos, sin [bs, num_key_value_heads, seq_len, head_dim] -> [bs, num_heads, seq_len, head_dim]
    n_repeat = q.shape[1] // cos.shape[1]
    q_embed = (q * cos.repeat_interleave(n_repeat, dim=1)) + (
        rotate_half(q) * sin.repeat_interleave(n_repeat, dim=1)
    )
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def monkey_patch_qwen3vl(rope_name, **kwargs):
    rope_index, rope_embed = get_multimodal_rope(rope_name, **kwargs)

    logging.info(f"Begin to patch Qwen3VLModel with {rope_name}.")
    Qwen3VLModel.get_rope_index = rope_index
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding = (
        rope_embed
    )

    if rope_name == "mhrope":
        # patch apply_rotary_embed
        transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb = (
            apply_multihead_rotary_pos_emb
        )
        logging.info(
            "MHRoPE: Patched apply_rotary_pos_emb with apply_multihead_rotary_pos_emb."
        )

    logging.info(f"Patched Qwen3VLModel with {rope_name}.")

    test_forward()

    if rope_name == "mhrope":
        transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb = (
            original_apply_rotary_pos_emb
        )
        logging.info("MHRoPE: Restored original apply_rotary_pos_emb.")


def test_forward():
    ckpt = "Qwen/Qwen3-VL-2B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        ckpt,
        attn_implementation="flash_attention_2",
        device_map="auto",
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(ckpt)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text)


if __name__ == "__main__":
    common_kwargs = dict(
        dim=128,
        base=5000000,
    )
    monkey_patch_qwen3vl("vanilla-rope", **common_kwargs)

    # MRoPE, MRoPE-I, MHRoPE
    monkey_patch_qwen3vl(
        "mrope", mrope_section=[16, 24, 24], temporal_stride=1, **common_kwargs
    )
    monkey_patch_qwen3vl(
        "mrope-interleave",
        mrope_section=[24, 20, 20],
        temporal_stride=1,
        spatial_reset=True,
        **common_kwargs,
    )
    monkey_patch_qwen3vl(
        "mhrope",
        num_key_value_heads=8,
        mrope_section=[2, 3, 3],
        temporal_stride=1,
        spatial_reset=True,
        **common_kwargs,
    )

    # VideoRoPE and HoPE, temporal_stride is a float
    monkey_patch_qwen3vl(
        "videorope", mrope_section=[16, 24, 24], temporal_stride=2.0, **common_kwargs
    )
    monkey_patch_qwen3vl(
        "hope",
        mrope_section=[16, 24, 24],
        temporal_stride=2.0,
        temporal_stride_lst=[0.5, 0.75, 1.0, 1.25, 1.5],
        **common_kwargs,
    )
