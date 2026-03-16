import itertools
from typing import Optional

import torch


def split_video_grid_thw(
    video_grid_thw: Optional[torch.LongTensor],
) -> Optional[torch.LongTensor]:
    """
    Qwen3-VL uses timestamps to separate videos (multiple segments per "video"),
    so we need to expand `video_grid_thw` accordingly (same logic as HF).
    """
    if video_grid_thw is None:
        return None
    video_grid_thw = torch.repeat_interleave(
        video_grid_thw, video_grid_thw[:, 0], dim=0
    )
    video_grid_thw[:, 0] = 1
    return video_grid_thw


def iter_mm_groups(mm_token_type_ids_1d: torch.Tensor):
    """
    Yield contiguous segments of the same modality:
    text (0), image (1), video (2).
    """
    for key, group in itertools.groupby(
        enumerate(mm_token_type_ids_1d.tolist()), lambda x: x[1]
    ):
        group = list(group)
        start_index = group[0][0]
        end_index = group[-1][0] + 1
        yield key, start_index, end_index
