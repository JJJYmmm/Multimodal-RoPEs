import transformers
from transformers import Qwen3VLModel, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from multimodal_ropes.entry import get_multimodal_rope

import logging
logging.basicConfig(level=logging.INFO)

def monkey_patch_qwen3vl(rope_name):
    rope_index, rope_embed = get_multimodal_rope(rope_name, dim=128, base=10000)
    Qwen3VLModel.get_rope_index = rope_index
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding = rope_embed

    logging.info(f"Patched Qwen3VLModel with {rope_name} RoPE.")
    # config = Qwen3VLConfig()
    # small_model = Qwen3VLForConditionalGeneration(config)

if __name__ == "__main__":
    monkey_patch_qwen3vl("vanilla-rope")