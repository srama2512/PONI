import numpy as np
import torch
import torch.nn.functional as F


def spatial_transform_map(p, x, invert=True, mode="bilinear"):
    """
    Inputs:
        p     - (bs, f, H, W) Tensor
        x     - (bs, 3) Tensor (x, y, theta) transforms to perform
    Outputs:
        p_trans - (bs, f, H, W) Tensor
    Conventions:
        Shift in X is rightward, and shift in Y is downward. Rotation is clockwise.
    Note: These denote transforms in an agent's position. Not the image directly.
    For example, if an agent is moving upward, then the map will be moving downward.
    To disable this behavior, set invert=False.
    """
    device = p.device
    H, W = p.shape[2:]

    trans_x = x[:, 0]
    trans_y = x[:, 1]
    # Convert translations to -1.0 to 1.0 range
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H / 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W / 2

    trans_x = trans_x / Wby2
    trans_y = trans_y / Hby2
    rot_t = x[:, 2]

    sin_t = torch.sin(rot_t)
    cos_t = torch.cos(rot_t)

    # This R convention means Y axis is downwards.
    A = torch.zeros(p.size(0), 3, 3).to(device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    A[:, 0, 2] = trans_x
    A[:, 1, 2] = trans_y
    A[:, 2, 2] = 1

    # Since this is a source to target mapping, and F.affine_grid expects
    # target to source mapping, we have to invert this for normal behavior.
    Ainv = torch.inverse(A)

    # If target to source mapping is required, invert is enabled and we invert
    # it again.
    if invert:
        Ainv = torch.inverse(Ainv)

    Ainv = Ainv[:, :2]
    grid = F.affine_grid(Ainv, p.size(), align_corners=False)
    p_trans = F.grid_sample(p, grid, mode=mode, align_corners=False)

    return p_trans


def crop_map(h, x, crop_size, mode="bilinear"):
    """
    Crops a tensor h centered around location x with size crop_size
    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer
    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(0)
        .expand(crop_size, -1)
        .contiguous()
        .float()
    )
    y_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(1)
        .expand(-1, crop_size)
        .contiguous()
        .float()
    )
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = center_grid.unsqueeze(0).expand(
        bs, -1, -1, -1
    )  # (bs, crop_size, crop_size, 2)
    crop_grid = crop_grid.contiguous()

    # Convert the grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
        crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
    ) / Wby2
    crop_grid[:, :, :, 1] = (
        crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
    ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode, align_corners=False)

    return h_cropped


def crop_map_with_pad(
    h, x, crop_size, mode="bilinear", pad_mode="constant", pad_value=0
):
    """
    Crops a tensor h centered around location x with size crop_size
    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer
    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    Note: if H != W, this pads "h" to get a square and and modifies "x" accordingly.
    """
    _, _, H, W = h.shape
    device = x.device
    if H > W:
        # Pad width
        D = H - W
        Dby2 = D // 2
        h = F.pad(h, (Dby2, D - Dby2, 0, 0), mode=pad_mode, value=pad_value)
        x = x + torch.Tensor([[Dby2, 0]]).float().to(device)
    elif W > H:
        # Pad height
        D = W - H
        Dby2 = D // 2
        h = F.pad(h, (0, 0, Dby2, D - Dby2), mode=pad_mode, value=pad_value)
        x = x + torch.Tensor([[0, Dby2]]).float().to(device)
    return crop_map(h, x, crop_size, mode=mode)


def subtract_poses(
    xyo_1: torch.Tensor,
    xyo_2: torch.Tensor,
):
    """
    Get xyo_2 in egocentric coordinates of xyo_1

    xyo_(1|2) - (bs, 3)

    Conventions:
    X is rightward, Y is downward. Theta is measured from X to Y.
    Origin does not matter as long as they're consistent across the two poses.
    """
    rel_xy = xyo_2[:, :2] - xyo_1[:, :2]
    rad = torch.norm(rel_xy, dim=1)
    phi = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])
    theta = phi - xyo_1[:, 2]
    rel_xyo = torch.stack(
        [
            rad * torch.cos(theta),
            rad * torch.sin(theta),
            xyo_2[:, 2] - xyo_1[:, 2],
        ],
        dim=1,
    )
    return rel_xyo


def get_frontiers_np(unexp_map: np.array, free_map: np.array):
    r"""
    Computes the map frontiers given unexplored and free spaces on the map.
    Works for numpy arrays. Reference:
    https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132

    Args:
        unexp_map - (H, W) int numpy array with 1 for unexplored cells, 0 o/w.
        free_map - (H, W) int numpy array with 1 for explored free cells, 0 o/w.

    Outputs:
        frontiers - (H, W) boolean numpy array
    """
    unexp_map_shiftup = np.pad(
        unexp_map, ((0, 1), (0, 0)), mode="constant", constant_values=0
    )[1:, :]
    unexp_map_shiftdown = np.pad(
        unexp_map, ((1, 0), (0, 0)), mode="constant", constant_values=0
    )[:-1, :]
    unexp_map_shiftleft = np.pad(
        unexp_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
    )[:, 1:]
    unexp_map_shiftright = np.pad(
        unexp_map, ((0, 0), (1, 0)), mode="constant", constant_values=0
    )[:, :-1]
    frontiers = (
        (free_map == unexp_map_shiftup)
        | (free_map == unexp_map_shiftdown)
        | (free_map == unexp_map_shiftleft)
        | (free_map == unexp_map_shiftright)
    ) & (
        free_map == 1
    )  # (H, W)

    return frontiers
