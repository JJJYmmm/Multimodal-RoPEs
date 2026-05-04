# RoPE Hyperparameters

This package exposes every variant through:

```python
from multimodal_ropes.entry import get_multimodal_rope

rope_index, rope_embed_cls = get_multimodal_rope("mrope-interleave", dim=128, base=5_000_000, ...)
```

The returned `rope_index` builds `position_ids` and `rope_deltas`; `rope_embed_cls`
builds the rotary `cos`/`sin` module used by the model.

## Shared Parameters

| Parameter | Applies to | Meaning | Typical value |
| --- | --- | --- | --- |
| `dim` | all variants | Variant-side rotary dimension used by position/frequency configs. Match the model attention head dimension unless a variant explicitly documents otherwise. | `128` for Qwen3-VL 2B-style configs |
| `base` | config metadata | Variant-side RoPE theta. Runtime rotary modules derive `inv_freq` from the standardized HF text config; keep this aligned with the model config for reproducible logs and custom utilities. | `5_000_000` for Qwen3-VL examples |

The HF text config still controls runtime details such as `head_dim`,
`partial_rotary_factor`, `rope_scaling`, and `rope_theta` after
`standardize_rope_params()`. The package parameters above are the variant-side
configuration used by `get_multimodal_rope`.

## Consistency Notes

The implementations below are adapted to a Qwen-style drop-in interface that
separates position design from rotary `cos/sin` generation. Some source papers
were written for different model families, so the package follows these rules:

| Variant | Source alignment | Interface adaptation |
| --- | --- | --- |
| `v2pe` | Follows the official V2PE visual stride rule, `rope_pos_id_stride / num_image_token`. | Uses Qwen vision token spans as the visual blocks and keeps standard 1D RoPE frequencies. |
| `ilrope` | Uses a reset visual layout and interleaved axis frequencies, matching the IL/Mogao-style design goal. | Uses the same Qwen-compatible visual block parsing as the other variants. |
| `omnirope` | Follows OmniGen2's `(shift, row, col)` position layout and per-block `shift += max(grid_extent)` idea. | Applies that layout to Qwen image/video token blocks rather than OmniGen2 diffusion image tokens. |
| `mmrope` | Implements Lumos-1's distributed `t,t,h,w,h,w,h,w` meta-component and optional RGB-space position scaling. | Uses Qwen/MRoPE-style position parsing before applying `position_scale`. |
| `grape` | Implements Appendix G's 2D/3D multiplicative GRAPE form and adds learned multimodal rotation-plane variants. | `canonical` is a drop-in `cos/sin` module; learned `axis`, `mixed`, and `block_mixed` modes rotate Q/K directly through `apply_qk`. |

## Variant Parameters

Each variant differs along two axes:

- **Position ID design**: how tokens are assigned one or more positional coordinates.
- **Frequency allocation**: how rotary feature pairs consume those coordinates.

In this repo, 3D variants usually build `position_ids` with shape
`[3, batch, seq_len]`, corresponding to `(t, h, w)`. 1D variants duplicate the
same scalar position across the three rows for compatibility with Qwen-style
RoPE call sites.

### `vanilla-rope`

No variant-specific parameters.

Use when you want a 1D text-like RoPE baseline. Visual tokens receive ordinary
monotonic sequence positions and no spatio-temporal structure.

Position ID design:

- Treat the full multimodal sequence as a flat 1D token stream.
- Text and visual tokens advance by one position per token.
- The same scalar position is copied to all three `(t, h, w)` rows.

Frequency allocation:

- Use standard 1D RoPE over the full rotary dimension.
- Only the first position row is used by the embedding module, so no channels are
  assigned separately to spatial or temporal axes.

```python
get_multimodal_rope("vanilla-rope", dim=128, base=5_000_000)
```

### `v2pe`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `visual_stride` | `16.0` | Total RoPE-position span assigned to one visual block. Each visual token advances by `visual_stride / num_visual_tokens`, matching the official V2PE `rope_pos_id_stride / num_image_token` rule. |

Use smaller `visual_stride` values to compress long visual blocks into a shorter
position range. For example, `visual_stride=1.0` makes a 4-token image occupy
positions `+0.25, +0.50, +0.75, +1.00` after the preceding token.

Position ID design:

- Treat the full multimodal sequence as a flat 1D token stream.
- Text tokens advance by one.
- Each visual block advances by `visual_stride / num_visual_tokens` per visual
  token; after the visual block, following text resumes from
  `ceil(last_visual_position) + 1`.
- The same scalar position is copied to all three `(t, h, w)` rows.

Frequency allocation:

- Same as vanilla 1D RoPE.
- V2PE changes only the visual token position spacing, not the rotary frequency
  layout.

```python
get_multimodal_rope("v2pe", dim=128, base=5_000_000, visual_stride=16.0)
```

### `mrope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[16, 24, 24]` | Positive rotary-pair allocation for `(t, h, w)` axes in chunked MRoPE frequency layout. Sum should match `head_dim // 2` for full RoPE. |
| `temporal_stride` | `2.0` | Positive multiplier applied to visual temporal indices. |

Use when reproducing Qwen2-VL/Qwen2.5-VL-style MRoPE.

Position ID design:

- Upcast visual tokens to `(t, h, w)` coordinates derived from
  `image_grid_thw` / `video_grid_thw` after `spatial_merge_size`.
- Text before, between, and after visual blocks uses ordinary 1D positions copied
  to all three rows.
- Visual positions are offset after preceding text, so visual and text positions
  live in one monotonically growing range.
- The temporal coordinate is multiplied by `temporal_stride`.

Frequency allocation:

- Split rotary pairs into chunked sections according to `mrope_section`.
- Chunks are consumed round-robin by `(t, h, w)` following the Qwen MRoPE layout:
  the first chunk uses `t`, the second `h`, the third `w`, then repeats if more
  chunks exist.

```python
get_multimodal_rope("mrope", dim=128, base=5_000_000, mrope_section=[16, 24, 24], temporal_stride=2.0)
```

### `mrope-interleave`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[24, 20, 20]` | Positive per-axis rotary-pair budget for interleaved frequency allocation. |
| `temporal_stride` | `1.0` | Positive multiplier applied to visual temporal indices. |
| `spatial_reset` | `False` | If `True`, reset visual `h/w` positions to local image/video coordinates and keep the visual end token after the local visual range. |

Use `spatial_reset=False` for Qwen3-VL/Qwen3.5-style compatibility; use
`spatial_reset=True` for the paper's spatial-reset experiments.

Position ID design:

- Uses the same Qwen-style 3D grid construction as MRoPE.
- With `spatial_reset=False`, visual `(t, h, w)` are all offset after preceding
  text, preserving a global monotonic range.
- With `spatial_reset=True`, visual `h/w` are local grid coordinates starting at
  zero, while `t` is still offset by preceding text. This decouples local spatial
  layout from global text positions.

Frequency allocation:

- Interleave frequencies across axes instead of assigning one contiguous band per
  axis.
- The implementation starts from the `t` frequency tensor and overwrites `h` and
  `w` positions at every third rotary pair, producing a fine-grained
  `t,h,w,t,h,w,...`-style allocation within each section.

```python
get_multimodal_rope("mrope-interleave", dim=128, base=5_000_000, mrope_section=[24, 20, 20], temporal_stride=1.0, spatial_reset=False)
```

### `mhrope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[2, 3, 3]` | Number of KV heads assigned to `(t, h, w)`. Unlike MRoPE, this is head allocation, not feature-pair allocation. The sum must not exceed `num_key_value_heads`. |
| `temporal_stride` | `1.0` | Positive multiplier applied to visual temporal indices. |
| `num_key_value_heads` | `8` | Total KV heads. If `sum(mrope_section) < num_key_value_heads`, remaining heads receive NoPE-style zero rotations. |
| `spatial_reset` | inherited default `False` | Passed through the MRoPE-I position design. |

MHRoPE changes the shape of `cos/sin` to include KV heads, so callers must also
patch the model's `apply_rotary_pos_emb` with the multi-head application helper
shown in the README.

Position ID design:

- Reuses the MRoPE-I position design, including the optional `spatial_reset`
  behavior.

Frequency allocation:

- Allocate entire KV heads, not feature chunks, to axes.
- `mrope_section=[2, 3, 3]` means 2 KV heads encode `t`, 3 encode `h`, and 3
  encode `w`.
- If fewer heads are assigned than `num_key_value_heads`, remaining heads get
  zero-frequency rotations, equivalent to NoPE for those heads.

```python
get_multimodal_rope("mhrope", dim=128, base=5_000_000, mrope_section=[2, 3, 3], num_key_value_heads=8, spatial_reset=True)
```

### `videorope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[16, 24, 24]` | Positive rotary-pair allocation used by the VideoRoPE frequency layout. |
| `temporal_stride` | `2.0` | Positive multiplier applied to visual temporal positions before the diagonal layout. |

VideoRoPE uses a diagonal visual position design and a frequency layout that
places spatial dimensions in alternating early channels while leaving temporal
information in lower-frequency channels.

Position ID design:

- Uses a diagonal visual layout.
- Visual temporal positions are offset by preceding text and multiplied by
  `temporal_stride`.
- Spatial coordinates are centered around the visual grid center, then added to
  the temporal coordinate. This couples space with time in a diagonal trajectory.

Frequency allocation:

- Spatial `h/w` axes occupy alternating early rotary pairs.
- Temporal information remains in the remaining lower-frequency pairs.
- This is intended to reduce MRoPE's high-frequency bias on the temporal axis.

```python
get_multimodal_rope("videorope", dim=128, base=5_000_000, mrope_section=[16, 24, 24], temporal_stride=2.0)
```

### `hope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[16, 24, 24]` | Same positive frequency allocation budget as VideoRoPE. |
| `temporal_stride` | `2.0` | Positive base temporal stride used by the HoPE position design. |
| `temporal_stride_lst` | `[0.5, 0.75, 1.0, 1.25, 1.5]` | Non-empty positive candidate temporal scaling values used by HoPE's positional scaling design. |

HoPE follows VideoRoPE frequency allocation but applies NoPE to the temporal
dimension in the embedding module.

Position ID design:

- Builds on VideoRoPE's diagonal visual layout.
- Adds HoPE positional scaling over temporal stride choices from
  `temporal_stride_lst`.
- This repo resets temporal positions before applying rotary frequencies, matching
  the local vectorized HoPE implementation.

Frequency allocation:

- Same spatial frequency allocation as VideoRoPE.
- The temporal axis is forced to zero in the embedding module, so temporal
  channels are effectively NoPE.

```python
get_multimodal_rope("hope", dim=128, base=5_000_000, mrope_section=[16, 24, 24], temporal_stride=2.0, temporal_stride_lst=[0.5, 0.75, 1.0, 1.25, 1.5])
```

### `circlerope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[16, 24, 24]` | Chunked 3D frequency allocation used after CircleRoPE position design. |
| `temporal_stride` | `1.0` | Temporal spacing for stacked video rings. |
| `move_to_origin` | `False` | Shift circle coordinates toward the origin. |
| `move_to_positive` | `False` | Shift circle coordinates into a positive range. |
| `dff_rate` | `False` | Distance/frequency interpolation weight in `[0, 1]`; `False` disables it and `True` is equivalent to `1.0`. |
| `method` | `"circle"` | Circle position construction mode. Options: `"circle"`, `"no_circle"`. |
| `radius` | `-1` | Circle radius; negative means use the implementation default. |
| `alpha` | `-1` | Circle layout coefficient in `[0, 1]`; `-1` means use the implementation default. |

The repo extends image-only CircleRoPE to video by stacking circular rings along
the temporal dimension.

Position ID design:

- Map each image's spatial grid onto a circular trajectory instead of a Cartesian
  `h/w` grid.
- For video, stack one circle/ring per temporal slice.
- `move_to_origin`, `move_to_positive`, `radius`, `alpha`, and `dff_rate` control
  how the circle is shifted, scaled, and distance-adjusted.

Frequency allocation:

- Uses the standard chunked MRoPE frequency allocation after constructing circular
  3D positions.
- In other words, CircleRoPE mainly changes position design, not the feature
  allocation rule.

```python
get_multimodal_rope("circlerope", dim=128, base=5_000_000, mrope_section=[16, 24, 24], temporal_stride=1.0, move_to_origin=True, dff_rate=True, radius=10, alpha=0.5)
```

### `ilrope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[24, 20, 20]` | Per-axis rotary-pair budget for interleaved 3D frequency allocation. |
| `temporal_stride` | `1.0` | Multiplier applied to visual temporal positions. |
| `spatial_reset` | `True` | Fixed to `True`; ILRoPE uses an Omni/Mogao-style reset layout in this package. Use `mrope-interleave` for the non-reset interleaved layout. |

ILRoPE uses the reset visual layout shared with OmniRoPE, then applies
MRoPE-Interleave frequency allocation.

Position ID design:

- Text tokens use identical positions on all three axes.
- Each visual block uses `(shift, row, col)`: temporal/sequence axis is fixed to
  the current block shift, while spatial axes reset to local row/column indices.
- After each visual block, `shift` advances by the maximum visual grid extent,
  including the temporal extent after `temporal_stride`.
- This gives local spatial reset behavior and intentionally differs from
  text-only RoPE compatibility on spatial rows.

Frequency allocation:

- Uses MRoPE-I interleaved frequency allocation.
- Compared with OmniRoPE, the position layout is the same in this repo, but the
  frequency allocation is interleaved rather than chunked.

```python
get_multimodal_rope("ilrope", dim=128, base=5_000_000, mrope_section=[24, 20, 20], temporal_stride=1.0)
```

### `omnirope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[16, 24, 24]` | Chunked 3D frequency allocation for `(t, h, w)`. |
| `temporal_stride` | `1.0` | Multiplier applied to visual temporal positions. |

OmniRoPE follows the OmniGen2-style position layout: text tokens use identical
positions on all three axes, while each visual block uses `(shift, row, col)` and
advances `shift` by the maximum visual grid extent.

Position ID design:

- Text tokens use identical positions on all three axes.
- Visual tokens use `(shift, row, col)`, where `shift` separates visual blocks and
  `row/col` reset locally for each block.
- `shift += max(t_extent, h, w)` after a visual block, where
  `t_extent=(t - 1) * temporal_stride + 1`. This keeps later text/visual blocks
  outside the temporally stretched visual range.

Frequency allocation:

- Uses chunked MRoPE allocation over `(t, h, w)`.
- The main difference from ILRoPE in this repo is therefore frequency allocation:
  OmniRoPE is chunked, ILRoPE is interleaved.

```python
get_multimodal_rope("omnirope", dim=128, base=5_000_000, mrope_section=[16, 24, 24], temporal_stride=1.0)
```

### `mmrope`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[16, 24, 24]` | Kept for API symmetry with 3D RoPE variants. The MM-RoPE embedding uses the fixed distributed meta-component pattern. |
| `temporal_stride` | `1.0` | Multiplier applied to visual temporal positions before scaling. |
| `position_scale` | `(1.0, 1.0, 1.0)` | Positive per-axis multiplier for `(t, h, w)` positions. Use values like `(4.0, 8.0, 8.0)` to scale latent coordinates toward RGB-space coordinates as in Lumos-1. |

MM-RoPE frequency allocation repeats the 16-channel meta component
`T,T,H,W,H,W,H,W` across rotary pairs.

Position ID design:

- Uses 3D visual coordinates offset after preceding text, similar to MRoPE.
- Applies per-axis scaling through `position_scale=(scale_t, scale_h, scale_w)`.
- For Lumos-1-style usage, use compression-ratio-like values such as
  `(4.0, 8.0, 8.0)` so latent video coordinates rotate closer to RGB-space
  distances.

Frequency allocation:

- Uses a fixed distributed meta-component repeated across rotary pairs:
  `T,T,H,W,H,W,H,W`.
- This keeps the original 2:3:3 temporal/spatial ratio within each 16-channel
  component while distributing all axes across the frequency spectrum.

```python
get_multimodal_rope("mmrope", dim=128, base=5_000_000, temporal_stride=1.0, position_scale=(4.0, 8.0, 8.0))
```

### `grape`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mrope_section` | `[16, 24, 24]` | Rotary-pair allocation for the commuting `(t, h, w)` GRAPE generators. |
| `temporal_stride` | `1.0` | Multiplier applied to visual temporal indices in the MRoPE-style 3D position design. |
| `log_freq_scale` | `1.0` | Optional positive scale applied to the standardized log-frequency spectrum before exponentiation. `1.0` recovers the RoPE-compatible multiplicative GRAPE baseline. |
| `frequency_allocation` | `"chunked"` | Fixed canonical-plane assignment used by `grape_mode="canonical"`. Options: `"chunked"`, `"interleaved"`. |
| `grape_mode` | `"canonical"` | GRAPE implementation mode. Options: `"canonical"`, `"axis"`, `"mixed"`, `"block_mixed"`. |
| `num_planes` | `None` | Number of learned rank-2 planes for `axis` and `mixed`. `None` means `rotary_dim // 2`, matching the number of ordinary RoPE pairs. |
| `block_size` | `16` | Local block width for `block_mixed`; `rotary_dim` must be divisible by it. Invalid block sizes raise instead of being silently clipped. |
| `planes_per_block` | `None` | Learned rank-2 planes per block for `block_mixed`. `None` means `block_size // 2`; values must not exceed `block_size // 2`. |
| `rope_init` | `True` | Initialize learned planes as ordinary even/odd RoPE pairs, and initialize mixed-axis weights in a cyclic `t,h,w` pattern. |
| `learnable` | `True` | Whether learned plane parameters and mixed-axis weights require gradients. |

`grape_mode="canonical"` is the drop-in path for existing Qwen/HF rotary call
sites: the module returns `cos/sin` and can be used like the other variants.
The learned modes follow the downloaded design note's multimodal GRAPE
adaptations and cannot be faithfully represented as one standalone `cos/sin`
tensor. They are trainable attention-module parameters and must be applied to
Q/K directly:

```python
_, rope_cls = get_multimodal_rope("grape", dim=128, base=5_000_000, grape_mode="block_mixed")
rotary_emb = rope_cls(text_config)
query_states, key_states = rotary_emb.apply_qk(query_states, key_states, position_ids)
```

Position ID design:

- Uses the MRoPE-style 3D position design by default, so visual tokens carry
  `(t, h, w)` coordinates.
- Text positions are copied to all three rows, as in other Qwen-compatible 3D
  variants.
- In `canonical` mode, if a caller directly supplies 2D position IDs, the
  embedding falls back to the 1D RoPE-compatible GRAPE path.
- Learned modes require 3D `position_ids` shaped `[3, batch, seq_len]` because
  the trainable phase directions consume explicit `(t, h, w)` coordinates.

Frequency allocation:

- For `canonical` 3D inputs, instantiates the paper's commuting multimodal GRAPE
  base design:
  `G3D(t,h,w)=exp(t Lt) exp(h Lh) exp(w Lw)` with canonical disjoint planes.
- `frequency_allocation="chunked"` assigns disjoint canonical planes according
  to `mrope_section`; with `log_freq_scale=1.0`, this is the MRoPE-compatible
  separable case.
- `frequency_allocation="interleaved"` assigns canonical planes in a repeating
  `t,h,w` pattern, preserving a broader frequency spectrum for every axis while
  keeping the generators commuting and disjoint.
- For 1D inputs, uses the canonical RoPE coordinate planes and log-uniform
  frequency spectrum when `log_freq_scale=1.0`.
- Learned `axis` mode implements a GRAPE-MRoPE adaptation:
  `G(t,h,w)=Gt(t) Gh(h) Gw(w)`, where each axis has its own learned rank-2 plane
  bank.
- Learned `mixed` mode gives every learned plane a trainable normalized
  `(alpha_t, alpha_h, alpha_w)` direction and uses the phase
  `theta_i * (alpha_t t + alpha_h h + alpha_w w)`.
- Learned `block_mixed` is the lower-cost engineering variant: it splits
  `rotary_dim` into local blocks, learns planes inside each block, and also
  learns a mixed `(t,h,w)` phase direction per block plane.
- In `block_mixed`, the frequency buffer is block-shaped
  `[num_blocks, planes_per_block]`, preserving each local block's original RoPE
  frequency band.
- `log_freq_scale` warps the standardized HF text-config frequency spectrum for
  both canonical and learned modes. Learned modes intentionally reuse
  `self.inv_freq`, so `rope_theta` and `partial_rotary_factor` stay aligned with
  the model config even if `base` is not repeated in `get_multimodal_rope`.

```python
get_multimodal_rope("grape", dim=128, base=5_000_000, mrope_section=[16, 24, 24], temporal_stride=1.0, log_freq_scale=1.0)
get_multimodal_rope("grape", dim=128, base=5_000_000, frequency_allocation="interleaved")
get_multimodal_rope("grape", dim=128, base=5_000_000, grape_mode="axis", num_planes=64)
get_multimodal_rope("grape", dim=128, base=5_000_000, grape_mode="mixed", num_planes=64)
get_multimodal_rope("grape", dim=128, base=5_000_000, grape_mode="block_mixed", block_size=16, planes_per_block=8)
```
