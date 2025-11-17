# Revisiting Multimodal Positional Encoding in Vision–Language Models

This repository is the official implementation of [Revisiting Multimodal Positional Encoding in Vision–Language Models](https://arxiv.org/abs/2510.23095). 

Multimodal position encoding is essential for vision-language models, yet there has been little systematic investigation into multimodal position encoding. We conduct a comprehensive analysis of *multimodal Rotary Positional Embedding (RoPE)* by examining its two core components: *position design* and *frequency allocation*. Through extensive experiments, we identify three key guidelines: *positional coherence, full frequency utilization, and preservation of textual priors*—ensuring unambiguous layout, rich representation, and faithful transfer from the pre-trained LLM. Based on these insights, we propose **Multi-Head RoPE (MHRoPE)** and **MRoPE-Interleave (MRoPE-I)**, two simple and plug-and-play variants that require no architectural changes. Our methods consistently outperform existing approaches across diverse benchmarks, with significant improvements in both general and fine-grained multimodal understanding.


<img width="42%" height="1475" alt="position-design" src="https://github.com/user-attachments/assets/25112335-b04e-4ff9-8203-ef6a787fa9ad" />
<img width="50%" height="548" alt="frequency-allocation" src="https://github.com/user-attachments/assets/43fd0a11-5a05-44f6-9685-8f83217c1410" />

## News

- 2025.10 All variants of [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) now adopt MRoPE-Interleave w/o *spatial-reset*!

## Todo List: Implementations of Multimodal RoPE Variants

To enhance usability and consistency, we are refactoring various multimodal RoPE implementations into a unified interface, similar to [`Qwen3VLTextRotaryEmbedding`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L278). This effort is expected to be completed within one week (target date: November 9, 2025).

- [x] [Vanilla RoPE](https://arxiv.org/abs/2104.09864): Apply vanilla RoPE directly to multimodal sequences, discarding the spatio-temporal structure of visual content, yet it remains a strong baseline.  
- [x] [MRoPE](https://arxiv.org/abs/2409.12191): Introduced in Qwen2VL, upcasts 1D positions to three axes (t, h, w) and splits the feature dimension across different positional axes.  
- [x] Our MRoPE-I: Applies an interleaved frequency allocation strategy, preserving a more complete frequency band for each positional axis; spatial reset is incorporated into the positional design to enhance visual attention.  
- [x] Our MHRoPE: Employs head-wise frequency allocation to maximize utility across attention heads; spatial reset is also used.  
- [x] [VideoRoPE](https://arxiv.org/pdf/2502.05173): Optimizes MRoPE’s frequency allocation by assigning the temporal dimension to low-frequency bands and adopting a diagonal positional design. Different from the official implementation, we vectorize the computation for faster execution.
- [x] [HoPE](https://arxiv.org/abs/2505.20444): Built on VideoRoPE, applies positional scaling in the design and uses NoPE (no positional encoding) on the temporal dimension during frequency allocation. Different from the official implementation, we reset the temporal positions and vectorize the computation for faster execution.
- [x] [CircleRoPE](https://arxiv.org/abs/2505.16416): A novel positional design that maps image tokens onto a circular trajectory orthogonal to text token indices, effectively mitigating cross-modal positional bias during generation. While the original CircleRoPE was designed for static images, we extend it to support video inputs by stacking circular rings along the temporal dimension, see [here](https://github.com/JJJYmmm/Multimodal-RoPEs/blob/64f8a141326c0ec079f8f05da42483a600028662/multimodal_ropes/pos_design/circlerope.py#L122-L130). Note that this repo doesn't support AGE mode for simplicity.
- [ ] [V2PE](https://arxiv.org/abs/2412.09616)
- [ ] [ILRoPE](https://arxiv.org/abs/2505.05472v1) / [OmniRoPE](https://arxiv.org/abs/2506.18871)
- [ ] [MMRoPE](https://arxiv.org/abs/2507.08801)
- [ ] More variants... Feel free to open an issue or pull request, and I will add them here. 🤗

## Usage

We organize various multimodal RoPE implementations under the `transformers` by decoupling them into two components: **position design** and **frequency allocation**.

### Installation

You can install the `multimodal-ropes` package directly from the repository:

```bash
git clone https://github.com/JJJYmmm/Multimodal-RoPEs.git
pip install -e .
# Successfully installed multimodal-ropes-0.1.0
```

### Integration with Vision–Language Models (e.g., Qwen3-VL)

The package provides a simple interface to plug in different multimodal RoPE variants. Below is an example of how to patch `Qwen3-VL` with your preferred RoPE configuration:
> Note that adapting to the new positional encodings typically requires additional training or fine-tuning to ensure optimal performance.

```python
from multimodal_ropes.entry import get_multimodal_rope

def rotate_half(x):
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_multihead_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies multi-head rotary positional embeddings for MHA/GQA.
    
    Args:
        q: [bs, num_heads, seq_len, head_dim]
        k: [bs, num_key_value_heads, seq_len, head_dim]
        cos, sin: [bs, num_key_value_heads, seq_len, head_dim] → broadcast to num_heads
    """
    n_repeat = q.shape[1] // cos.shape[1]
    q_embed = (q * cos.repeat_interleave(n_repeat, dim=1)) + (rotate_half(q) * sin.repeat_interleave(n_repeat, dim=1))
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def monkey_patch_qwen3vl(rope_name, **kwargs):
    rope_index, rope_embed = get_multimodal_rope(rope_name, **kwargs)

    logging.info(f"Begin patching Qwen3VLModel with {rope_name}.")
    Qwen3VLModel.get_rope_index = rope_index
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding = rope_embed

    if rope_name == "mhrope":
        # Special handling for MHRoPE: replace the rotary embedding application function
        transformers.models.qwen3_vl.modeling_qwen3_vl.apply_rotary_pos_emb = apply_multihead_rotary_pos_emb
        logging.info("MHRoPE: Replaced apply_rotary_pos_emb with multi-head version.")

    logging.info(f"Successfully patched Qwen3VLModel with {rope_name}.")
    test_forward()  # Optional: run a minimal forward pass to verify

# Common RoPE configuration
common_kwargs = dict(
    dim=128,
    base=5_000_000,
)

# Examples of patching with different RoPE variants
monkey_patch_qwen3vl("vanilla-rope", **common_kwargs)
monkey_patch_qwen3vl("mrope-interleave", mrope_section=[24, 20, 20], temporal_stride=1, spatial_reset=True, **common_kwargs)
monkey_patch_qwen3vl("mhrope", num_key_value_heads=8, mrope_section=[2, 3, 3], temporal_stride=1, spatial_reset=True, **common_kwargs)
```

For more comprehensive examples and testing utilities, please refer to [`test.py`](test.py).

## Citation

If you find this repository is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@misc{huang2025revisitingmultimodalpositionalencoding,
      title={Revisiting Multimodal Positional Encoding in Vision-Language Models}, 
      author={Jie Huang and Xuejing Liu and Sibo Song and Ruibing Hou and Hong Chang and Junyang Lin and Shuai Bai},
      journal={arXiv preprint arXiv:2510.23095},
      year={2025}
}
```

## License

The content of this project itself is licensed under [LICENSE](LICENSE).
