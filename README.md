# Revisiting Multimodal Positional Encoding in Vision–Language Models

This repository is the official implementation of [Revisiting Multimodal Positional Encoding in Vision–Language Models](https://arxiv.org/abs/2510.23095). 

Multimodal position encoding is essential for vision-language models, yet there has been little systematic investigation into multimodal position encoding. We conduct a comprehensive analysis of *multimodal Rotary Positional Embedding (RoPE)* by examining its two core components: *position design* and *frequency allocation*. Through extensive experiments, we identify three key guidelines: *positional coherence, full frequency utilization, and preservation of textual priors*—ensuring unambiguous layout, rich representation, and faithful transfer from the pre-trained LLM. Based on these insights, we propose **Multi-Head RoPE (MHRoPE)** and **MRoPE-Interleave (MRoPE-I)**, two simple and plug-and-play variants that require no architectural changes. Our methods consistently outperform existing approaches across diverse benchmarks, with significant improvements in both general and fine-grained multimodal understanding.
## News

- 2025.10 All variants of [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) now adopt MRoPE-Interleave w/o *spatial-reset*!

## To-Do List: Implementations of Multimodal RoPE Variants

To enhance usability and consistency, we are refactoring various multimodal RoPE implementations into a unified interface, similar to [`Qwen3VLTextRotaryEmbedding`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L278). This effort is expected to be completed within one week (target date: November 9, 2025).

- [ ] Our MHRoPE / MRoPE-I
- [ ] MRoPE / CircleRoPE
- [ ] VideoRoPE / HoPE
- [ ] ILRoPE / OmniRoPE
- [ ] MMRoPE
- [ ] More variants...

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
