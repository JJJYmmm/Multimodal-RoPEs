import math
import torch
from typing import Optional

from ..configs.circlerope import CircleRopeConfig


def get_circlerope_index(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    extra_config: CircleRopeConfig = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # qwen3vl use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    # if you are using qwen2/2.5vl, please remove them
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
            # dtype=input_ids.dtype,
            dtype=torch.float,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
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
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .view(-1, llm_grid_h, llm_grid_w)
                ) * extra_config.temporal_stride
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                )

                llm_pos_ids = _circle_projection(
                    w_index, h_index, t_index, extra_config
                )

                # original circle rope only supports image input, we extend it to support video input
                # by increasing the time dimension linearly
                llm_pos_ids = (
                    llm_pos_ids.repeat(1, llm_grid_t)
                    + torch.arange(llm_grid_t)
                    .view(1, -1)
                    .repeat_interleave(llm_grid_h * llm_grid_w, dim=1)
                    * extra_config.temporal_stride
                )

                llm_pos_ids_list.append(llm_pos_ids + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


# modified from https://github.com/lose4578/CircleRoPE
def _circle_projection(
    w_index, h_index, t_index, extra_config: CircleRopeConfig = None
):
    # Load circle rope configurations
    move_to_origin = extra_config.move_to_origin
    move_to_positive = extra_config.move_to_positive
    dff_rate = extra_config.dff_rate
    method = extra_config.method
    radius = extra_config.radius
    alpha = extra_config.alpha

    # Stack original coordinates
    ori_coords = torch.stack((w_index, h_index, t_index), dim=0)
    if move_to_origin:
        # Move coordinates to origin if specified
        ori_coords = move_to_origin_coords(ori_coords)

    # Determine radius: auto or fixed value
    if "auto" in str(radius):
        if radius == "auto":
            radius_scale = 1
        else:
            _, radius_scale = radius.split("-")
        # Calculate radius based on the maximum absolute coordinate value
        radius = ori_coords.max().abs() * float(radius_scale)
    else:
        radius = float(radius)

    # Perform circle projection
    convert_coords = circle_projection(
        ori_coords, text_vector=[1, 1, 1], radius=radius, alpha=alpha, method=method
    )

    # Apply differential rate if specified
    if dff_rate:
        no_circle_convert_coords = circle_projection(
            ori_coords, text_vector=[1, 1, 1], radius=-1, alpha=-1, method="no_circle"
        )
        # Linearly interpolate between circle projection and original coordinates
        convert_coords = (
            1 - dff_rate
        ) * convert_coords + dff_rate * no_circle_convert_coords

    # Move coordinates to positive axis if specified
    if move_to_positive:
        if move_to_positive == "auto":
            offset = 0
        else:
            offset = float(move_to_positive)
        convert_coords = move_to_positive_axis(convert_coords, offset=offset)

    # Flatten coordinate dimensions
    w_index = convert_coords[0].flatten()
    h_index = convert_coords[1].flatten()
    t_index = convert_coords[2].flatten()

    # Stack coordinates for language model position IDs
    llm_pos_ids = torch.stack([t_index, h_index, w_index])

    return llm_pos_ids


def move_to_origin_coords(coords):
    """
    Moves the center of the cube to the origin (stacked coordinates version).
    Parameters:
        coords: Tensor of shape (3, depth, height, width)
                Channel order corresponds to [x, y, z] axis coordinates.
    Returns:
        new_coords: Center-aligned coordinate tensor, maintaining the same shape.
    """
    # Calculate the center point for each axis [x_center, y_center, z_center]
    max_vals = torch.amax(
        coords, dim=(1, 2, 3)
    )  # Get maximum value along spatial dimensions
    min_vals = torch.amin(
        coords, dim=(1, 2, 3)
    )  # Get minimum value along spatial dimensions
    centers = (max_vals + min_vals) / 2.0
    # Adjust dimensions for broadcast subtraction (3, 1, 1, 1)
    centers = centers.view(-1, 1, 1, 1)
    # Perform translation
    new_coords = coords - centers

    return new_coords


def move_to_positive_axis(coords, offset=0):
    # Find the absolute minimum value across all coordinates
    min_vals = torch.abs(torch.min(coords))
    # Create a tensor of these minimum values for shifting
    centers = torch.tensor([min_vals, min_vals, min_vals]).view(-1, 1, 1, 1)

    # Shift coordinates to be positive and add an optional offset
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

    # Original non-linear circular mapping
    if method == "circle":
        coord_circle = map_to_circle(coords, radius, alpha)
    elif method == "no_circle":
        # Pass through coordinates if no circle mapping is specified
        coord_circle = coords
    else:
        raise ValueError(f"Invalid circle projection method: {method}")

    if rotate:
        # Rotate the plane to be perpendicular to the text_vector
        coord_circle = rotate_plane_perpendicular_to_vector(coord_circle, text_vector)

    return coord_circle


def rotate_plane_perpendicular_to_vector(coord_circle, text_vector):
    data_device = coord_circle.device
    data_dtype = coord_circle.dtype

    # Construct the target plane coordinate system
    text = torch.tensor(text_vector, dtype=data_dtype, device=data_device).float()
    text_norm = torch.norm(text)
    if text_norm < 1e-6:
        raise ValueError("text_vector cannot be zero vector")
    text_unit = text / text_norm  # Normalize the text vector

    # Construct an orthogonal basis
    if torch.abs(text_unit[0]) < 1e-6 and torch.abs(text_unit[1]) < 1e-6:
        # Handle the case where the vector is along the z-axis
        u = torch.tensor([1.0, 0.0, 0.0], device=data_device, dtype=data_dtype)
        v = torch.tensor([0.0, 1.0, 0.0], device=data_device, dtype=data_dtype)
    else:
        # Construct the first orthogonal vector u
        u = torch.stack(
            [-text_unit[1], text_unit[0], torch.tensor(0.0, device=data_device)]
        )
        u = u / torch.norm(u)  # Normalize u
        # Construct the second orthogonal vector v using cross product
        v = torch.cross(text_unit, u, dim=0)
        v = v / torch.norm(v)  # Normalize v

    # Project the circle points onto the new coordinate system
    x_components = (
        coord_circle[0] * u[0] + coord_circle[1] * v[0]
    )  # Contribution to new X from original X and Y
    y_components = (
        coord_circle[0] * u[1] + coord_circle[1] * v[1]
    )  # Contribution to new Y from original X and Y
    z_components = (
        coord_circle[0] * u[2] + coord_circle[1] * v[2]
    )  # Contribution to new Z from original X and Y

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
    # Extract x, y, z components; here x and y are tensors of shape (H, W)
    x = tensor[0, 0]
    y = tensor[1, 0]
    z = tensor[2, 0]  # z coordinates are preserved
    H, W = x.shape

    def get_norm_theta():
        # Method 1: Calculate angle using original coordinates, then linearly normalize to [0, 2π]
        theta_orig = torch.atan2(y, x)
        theta_min = theta_orig.min()
        theta_max = theta_orig.max()
        theta_range = theta_max - theta_min
        if theta_range > 0:
            theta_uniform = (theta_orig - theta_min) / theta_range * (2 * math.pi)
        else:
            # Handle cases with a single point or all points collinear through origin
            theta_uniform = theta_orig
        return theta_uniform

    def get_index_theta():
        # Method 2: Generate uniformly distributed angles based on grid indices
        indices = torch.arange(
            H * W, dtype=torch.float32, device=tensor.device
        ).reshape(H, W)
        theta_uniform = indices / (H * W) * (2 * math.pi)
        return theta_uniform

    # The larger alpha is, the closer it is to the normalized original angle.
    # When alpha=0, the grid index method is fully used.
    # When alpha=1, the original coordinate calculation angle is fully used.
    theta_norm = get_norm_theta()
    theta_index = get_index_theta()
    # Combine the two methods for calculating theta based on alpha
    theta_uniform = alpha * theta_norm + (1 - alpha) * theta_index

    # Generate mapped x, y coordinates based on the calculated uniform angle
    new_x = radius * torch.cos(theta_uniform)
    new_y = radius * torch.sin(theta_uniform)

    # Combine the three channels and maintain the shape as (3, 1, H, W)
    new_tensor = torch.stack([new_x, new_y, z], dim=0).unsqueeze(1)

    return new_tensor
