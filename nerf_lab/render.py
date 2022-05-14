import typing as t

import torch

import nerf_lab.utils as ut


def get_ray_bundle(
    height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor
):
    """Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
        height (int): Height of an image (number of pixels).
        width (int): Width of an image (number of pixels).
        focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated instinsics).
        tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
            transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
        ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
            each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
            row index `j` and column index `i`.
        ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
            direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
            passing through the pixel at row index `j` and column index `i`.
    """
    ii, jj = ut.meshgrid_xy(
        torch.arange(width).to(tform_cam2world),
        torch.arange(height).to(tform_cam2world),
    )
    directions = torch.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def compute_query_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: t.Optional[bool] = True,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 3D points are to be sampled.

    Args:
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
            `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
            `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
            coordinate that is of interest/relevance).
        far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
            coordinate that is of interest/relevance).
        num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
            randomly, whilst trying to ensure "some form of" uniform spacing among them.
        randomize (optional, bool): Whether or not to randomize the sampling of query points.
            By default, this is set to `True`. If disabled (by setting to `False`), we sample
            uniformly spaced points along each ray in the "bundle".

    Returns:
        query_points (torch.Tensor): Query points along each ray
            (shape: :math:`(width, height, num_samples, 3)`).
        depth_values (torch.Tensor): Sampled depth values along each ray
            (shape: :math:`(num_samples)`).
    """
    # fmt: off
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize is True:
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = (depth_values + torch.rand(noise_shape).to(ray_origins) * (far_thresh - near_thresh) / num_samples)

    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    # fmt: on

    return query_points, depth_values


def render_volume_density(
    radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
        radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
            we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
            the paper) (shape: :math:`(width, height, num_samples, 4)`).
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
            `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
            depth_values (torch.Tensor): Sampled depth values along each ray
            (shape: :math:`(num_samples)`).

    Returns:
        rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
        depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
        acc_map (torch.Tensor): Accumulated transmittance map).
    """
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * ut.cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map
