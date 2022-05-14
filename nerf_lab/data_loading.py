import typing as t

import torch


def positional_encoding(
    tensor: torch.Tensor,
    num_encoding_functions: int = 6,
    include_input: bool = True,
    log_sampling: bool = True,
) -> torch.Tensor:
    """Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        num_encoding_functions (optional, int): Number of encoding functions used to
            compute a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            computed positional encoding (default: True).
        log_sampling (optional, bool): Sample logarithmically in frequency space, as
            opposed to linearly (default: True).

    Returns:
        (torch.Tensor): Positional encoding of the input tensor.
    """
    encoding = [tensor] if include_input else []

    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    return encoding[0] if len(encoding) == 1 else torch.cat(encoding, dim=-1)


def get_minibatches(
    ray_bundle: torch.Tensor, chunksize: t.Optional[int] = 1024 * 8
) -> t.List[torch.Tensor]:
    """Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
        Each element of the list (except possibly the last) has dimension `0` of length
        `chunksize`.

    Args:
        inputs (torch.Tensor): Ray bundle.
        chunksize (t.Optional[int], optional): Chunk size. Defaults to 1024*8.

    Returns:
        t.List[torch.Tensor]: List of mini batches created based on ray bundle.
    """
    return [
        ray_bundle[i : i + chunksize] for i in range(0, ray_bundle.shape[0], chunksize)
    ]
