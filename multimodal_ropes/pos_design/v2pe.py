import math
from typing import Optional

import torch

from ..configs.v2pe import V2PEConfig


def get_v2pe_index(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    extra_config: V2PEConfig = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.float,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            last_record_pos_id = -1.0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                text_len = ed - st
                if text_len > 0:
                    text_pos = (
                        torch.arange(text_len, device=input_ids.device).float()
                        + last_record_pos_id
                        + 1
                    )
                    llm_pos_ids_list.append(text_pos.view(1, -1).expand(3, -1))
                    last_record_pos_id = float(text_pos[-1].item())

                visual_len = (
                    t.item()
                    * (h.item() // spatial_merge_size)
                    * (w.item() // spatial_merge_size)
                )
                small_stride = float(extra_config.visual_stride) / visual_len
                visual_pos = torch.arange(
                    last_record_pos_id + small_stride,
                    last_record_pos_id + small_stride * (visual_len + 1),
                    small_stride,
                    device=input_ids.device,
                    dtype=torch.float,
                )[:visual_len]
                llm_pos_ids_list.append(visual_pos.view(1, -1).expand(3, -1))
                last_record_pos_id = float(math.ceil(visual_pos[-1].item()))
                st = ed + visual_len

            if st < len(input_tokens):
                text_len = len(input_tokens) - st
                text_pos = (
                    torch.arange(text_len, device=input_ids.device).float()
                    + last_record_pos_id
                    + 1
                )
                llm_pos_ids_list.append(text_pos.view(1, -1).expand(3, -1))

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                device=position_ids.device, dtype=position_ids.dtype
            )
            mrope_position_deltas.append(
                torch.ceil(llm_positions.max()) + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(torch.float)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[
            0
        ]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
            .to(torch.float)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1], device=input_ids.device, dtype=torch.float
        )

    return position_ids, mrope_position_deltas
