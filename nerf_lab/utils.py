import typing as t

import torch


def meshgrid_xy(
    f_tensor: torch.Tensor, s_tensor: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
        f_tensor (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
        s_tensor (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    ii, jj = torch.meshgrid(f_tensor, s_tensor)

    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
        tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.

    Returns:
        cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
            tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    cumprod = torch.cumprod(tensor, dim=-1)
    cumprod = torch.roll(cumprod, 1, dim=-1)
    cumprod[..., 0] = 1.0

    return cumprod
