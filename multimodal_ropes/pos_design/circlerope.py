import math
from typing import Optional

import torch

from ..configs.circlerope import CircleRopeConfig
from ._qwen3vl import iter_mm_groups, split_video_grid_thw


def get_circlerope_index(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    extra_config: CircleRopeConfig = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    transformers==5.3.0: use `mm_token_type_ids` for modality grouping, but keep the original
    CircleRoPE vision position design (circle projection) to preserve legacy behavior.
    """
    if input_ids is None or mm_token_type_ids is None:
        raise ValueError(
            "Qwen3-VL get_rope_index requires `input_ids` and `mm_token_type_ids` in transformers==5.3.0"
        )

    video_grid_thw = split_video_grid_thw(video_grid_thw)
    spatial_merge_size = self.config.vision_config.spatial_merge_size

    # Legacy CircleRoPE uses float position ids (projection can be non-integer).
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=torch.float,
        device=input_ids.device,
    )
    mrope_position_deltas = []

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

        current_pos = 0.0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in iter_mm_groups(input_token_type):
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device, dtype=torch.float)
                    .view(1, -1)
                    .expand(3, -1)
                    + current_pos
                )
                current_pos += float(text_len)
                continue

            grid_thw = next(grid_iters[modality_type])
            llm_grid_t, llm_grid_h, llm_grid_w = (
                int(grid_thw[0].item()),
                int(grid_thw[1].item()) // spatial_merge_size,
                int(grid_thw[2].item()) // spatial_merge_size,
            )

            temporal_stride = 1.0
            if extra_config is not None:
                temporal_stride = float(extra_config.temporal_stride)

            # Original CircleRoPE builds a (T,H,W) grid then projects.
            t_index = (
                torch.arange(llm_grid_t, device=input_ids.device, dtype=torch.float)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .view(-1, llm_grid_h, llm_grid_w)
            ) * temporal_stride
            h_index = (
                torch.arange(llm_grid_h, device=input_ids.device, dtype=torch.float)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
            )
            w_index = (
                torch.arange(llm_grid_w, device=input_ids.device, dtype=torch.float)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
            )

            llm_pos_ids = _circle_projection(w_index, h_index, t_index, extra_config)

            # Extend to videos by increasing the time dimension linearly (kept from legacy impl).
            llm_pos_ids = llm_pos_ids.repeat(1, llm_grid_t) + (
                torch.arange(llm_grid_t, device=input_ids.device, dtype=torch.float)
                .view(1, -1)
                .repeat_interleave(llm_grid_h * llm_grid_w, dim=1)
                * temporal_stride
            )

            llm_pos_ids_list.append(llm_pos_ids + current_pos)
            current_pos = float((llm_pos_ids + current_pos).max().item() + 1.0)

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


# modified from https://github.com/lose4578/CircleRoPE
def _circle_projection(
    w_index, h_index, t_index, extra_config: CircleRopeConfig | None = None
):
    move_to_origin = extra_config.move_to_origin
    move_to_positive = extra_config.move_to_positive
    dff_rate = extra_config.dff_rate
    method = extra_config.method
    radius = extra_config.radius
    alpha = extra_config.alpha

    ori_coords = torch.stack((w_index, h_index, t_index), dim=0)
    if move_to_origin:
        ori_coords = move_to_origin_coords(ori_coords)

    # Determine radius: auto or fixed value
    if "auto" in str(radius):
        if radius == "auto":
            radius_scale = 1
        else:
            _, radius_scale = str(radius).split("-")
        radius = ori_coords.max().abs() * float(radius_scale)
    else:
        radius = float(radius)

    convert_coords = circle_projection(
        ori_coords, text_vector=[1, 1, 1], radius=radius, alpha=alpha, method=method
    )

    if dff_rate:
        no_circle_convert_coords = circle_projection(
            ori_coords, text_vector=[1, 1, 1], radius=-1, alpha=-1, method="no_circle"
        )
        convert_coords = (
            1 - dff_rate
        ) * convert_coords + dff_rate * no_circle_convert_coords

    if move_to_positive:
        if move_to_positive == "auto":
            offset = 0
        else:
            offset = float(move_to_positive)
        convert_coords = move_to_positive_axis(convert_coords, offset=offset)

    w_index = convert_coords[0].flatten()
    h_index = convert_coords[1].flatten()
    t_index = convert_coords[2].flatten()

    return torch.stack([t_index, h_index, w_index])


def move_to_origin_coords(coords):
    """
    Moves the center of the cube to the origin (stacked coordinates version).
    Parameters:
        coords: Tensor of shape (3, depth, height, width)
                Channel order corresponds to [x, y, z] axis coordinates.
    Returns:
        new_coords: Center-aligned coordinate tensor, maintaining the same shape.
    """
    max_vals = torch.amax(coords, dim=(1, 2, 3))
    min_vals = torch.amin(coords, dim=(1, 2, 3))
    centers = (max_vals + min_vals) / 2.0
    centers = centers.view(-1, 1, 1, 1)
    new_coords = coords - centers
    return new_coords


def move_to_positive_axis(coords, offset=0):
    min_vals = torch.abs(torch.min(coords))
    centers = torch.tensor([min_vals, min_vals, min_vals]).view(-1, 1, 1, 1)
    new_coords = coords + centers + offset
    return new_coords


def circle_projection(
    coords, text_vector=[1, 1, 1], radius=1.0, alpha=0.5, method="circle", rotate=True
):
    """
    Maps a point cloud to the circumference of a circle on a plane perpendicular to the given text_vector.
    Parameters:
        coords: [3, N] or [3, D, H, W] point cloud or stacked coordinates.
        text_vector: [3] Normal vector of the target plane.
        radius: Target circle radius.
        alpha: Nonlinear coefficient (0-1, controls distribution density).
        method: 'circle' for mapping to circle, 'no_circle' for no mapping.
        rotate: Boolean, whether to rotate the plane.
    """
    if method == "circle":
        coord_circle = map_to_circle(coords, radius, alpha)
    elif method == "no_circle":
        coord_circle = coords
    else:
        raise ValueError(f"Invalid circle projection method: {method}")

    if rotate:
        coord_circle = rotate_plane_perpendicular_to_vector(coord_circle, text_vector)

    return coord_circle


def rotate_plane_perpendicular_to_vector(coord_circle, text_vector):
    data_device = coord_circle.device
    data_dtype = coord_circle.dtype

    text = torch.tensor(text_vector, dtype=data_dtype, device=data_device).float()
    text_norm = torch.norm(text)
    if text_norm < 1e-6:
        raise ValueError("text_vector cannot be zero vector")
    text_unit = text / text_norm

    if torch.abs(text_unit[0]) < 1e-6 and torch.abs(text_unit[1]) < 1e-6:
        u = torch.tensor([1.0, 0.0, 0.0], device=data_device, dtype=data_dtype)
        v = torch.tensor([0.0, 1.0, 0.0], device=data_device, dtype=data_dtype)
    else:
        u = torch.stack(
            [-text_unit[1], text_unit[0], torch.tensor(0.0, device=data_device)]
        )
        u = u / torch.norm(u)
        v = torch.cross(text_unit, u, dim=0)
        v = v / torch.norm(v)

    x_components = coord_circle[0] * u[0] + coord_circle[1] * v[0]
    y_components = coord_circle[0] * u[1] + coord_circle[1] * v[1]
    z_components = coord_circle[0] * u[2] + coord_circle[1] * v[2]

    coord_componets = torch.stack([x_components, y_components, z_components])
    return coord_componets


def map_to_circle(tensor, radius=1.0, alpha=0.5):
    """
    Maps points on a plane (z coordinate is 0) to the edge of a circle centered at (0, 0, 0) with the given radius.

    Parameters:
        tensor: A tensor of shape (3, 1, H, W), where the three channels are x, y, z coordinates (here z coordinates are all 0).
        radius: The radius of the mapped circle, default 1.0.
        alpha: Value range [0,1], represents the weight of the normalized original angle; default 0.5.

    Returns:
        A tensor of the same shape as the input tensor, where each point on the plane is mapped to the edge of the circle, and the z coordinate remains unchanged.
    """
    x = tensor[0, 0]
    y = tensor[1, 0]
    z = tensor[2, 0]
    h, w = x.shape

    def get_norm_theta():
        theta_orig = torch.atan2(y, x)
        theta_min = theta_orig.min()
        theta_max = theta_orig.max()
        theta_range = theta_max - theta_min
        if theta_range > 0:
            theta_uniform = (theta_orig - theta_min) / theta_range * (2 * math.pi)
        else:
            theta_uniform = theta_orig
        return theta_uniform

    def get_index_theta():
        indices = torch.arange(
            h * w, dtype=torch.float32, device=tensor.device
        ).reshape(h, w)
        theta_uniform = indices / (h * w) * (2 * math.pi)
        return theta_uniform

    theta_norm = get_norm_theta()
    theta_index = get_index_theta()
    theta_uniform = alpha * theta_norm + (1 - alpha) * theta_index

    new_x = radius * torch.cos(theta_uniform)
    new_y = radius * torch.sin(theta_uniform)
    new_tensor = torch.stack([new_x, new_y, z], dim=0).unsqueeze(1)
    return new_tensor
