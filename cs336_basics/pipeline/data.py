from collections.abc import Generator
from typing import TypeAlias

from jaxtyping import Int
import numpy as np
import torch

from cs336_basics.common import constants as C

pair_type: TypeAlias = tuple[
    Int[torch.Tensor, "batch_size context_length"],
    Int[torch.Tensor, "batch_size context_length"],
]


def data_loading(
    x: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: str | None = None,
    **kwargs,
) -> Generator[pair_type] | pair_type:
    is_mem_efficient = kwargs.get("is_mem_efficient", False)
    if is_mem_efficient:
        return _data_loading_to_generator(
            x, batch_size=batch_size, context_length=context_length, device=device
        )
    else:
        return _data_loading(
            x, batch_size=batch_size, context_length=context_length, device=device
        )


def _data_loading_to_generator(
    x: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: str | None = None,
) -> Generator[pair_type]:
    """
    Given a numpy array of shape (n_tokens,),
    sample a batch of input sequences and next-token targets.

    Args:
        x (np.ndarray):
            The input array of shape (n_tokens,).
        batch_size (int):
            The number of samples to draw from the input array.
        context_length (int):
            The length of each input sequence.
        device (str | None, optional):
            The device to use for the output tensors. Defaults to CPU.

    Returns:
        A tuple containing the sampled input sequences and next-token targets.
    """
    device = device or C.DEFAULT_DEVICE
    target_offset = 1

    sample_indices, target_indices = _sample_indices(
        len(x), batch_size, context_length, target_offset
    )

    yield (
        torch.from_numpy(x[sample_indices]).to(device),
        torch.from_numpy(x[target_indices]).to(device),
    )


def _data_loading(
    x: np.ndarray, *, batch_size: int, context_length: int, device: str | None = None
) -> pair_type:
    device = device or C.DEFAULT_DEVICE
    target_offset = 1

    sample_indices, target_indices = _sample_indices(
        len(x), batch_size, context_length, target_offset
    )

    return torch.from_numpy(x[sample_indices]).to(device), torch.from_numpy(
        x[target_indices]
    ).to(device)


def _sample_indices(
    total_length: int, batch_size: int, context_length: int, target_offset: int
) -> pair_type:
    # Attention: randint is right exclusive
    # the largest value by randint(0, L - C) is (L - C - 1)
    # it handles the offset 1 for target automatically
    start_indices = np.random.randint(0, total_length - context_length, batch_size)
    start_indices = start_indices[:, None]  # (B, ) -> (B, 1)
    token_indices_in_batch = np.arange(context_length)  # (C, )
    sample_indices = start_indices + token_indices_in_batch  # (B, C)
    target_indices = sample_indices + target_offset

    return sample_indices, target_indices
