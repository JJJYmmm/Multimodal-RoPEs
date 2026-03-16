import itertools
import torch
from typing import Optional

from ..configs.vanilla import VanillaRopeConfig


def get_vanilla_rope_index(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    extra_config: VanillaRopeConfig = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    transformers==5.3.0 Qwen3-VL expects get_rope_index to use `mm_token_type_ids` for modality grouping:
    text (0), image (1), video (2).
    """
    if input_ids is None or mm_token_type_ids is None:
        raise ValueError(
            "Qwen3-VL get_rope_index requires `input_ids` and `mm_token_type_ids` in transformers==5.3.0"
        )

    # Separate video grids because Qwen3-VL uses timestamps to separate videos.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    spatial_merge_size = self.config.vision_config.spatial_merge_size

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            keep = attention_mask[batch_idx].bool()
            current_input_ids = current_input_ids[keep]
            input_token_type = input_token_type[keep]

        input_type_group = []
        for key, group in itertools.groupby(
            enumerate(input_token_type.tolist()), lambda x: x[1]
        ):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device)
                    .view(1, -1)
                    .expand(3, -1)
                    + current_pos
                )
                current_pos += text_len
            else:
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = self.get_vision_position_ids(
                    current_pos,
                    grid_thw,
                    1,
                    spatial_merge_size,
                    device=input_ids.device,
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                llm_positions.to(position_ids.device)
            )
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)

        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas
