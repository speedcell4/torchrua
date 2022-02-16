from typing import Tuple

import torch
from torch import Tensor
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_ptr
from torchrua.utils import accumulate_sizes

__all__ = [
    'head_catted_indices', 'head_catted_sequence',
    'last_catted_indices', 'last_catted_sequence',
    'init_catted_indices', 'init_catted_sequence',
    'tail_catted_indices', 'tail_catted_sequence',
]


@torch.no_grad()
def head_catted_indices(token_sizes: Tensor, device: Device = None) -> Tensor:
    if device is None:
        device = token_sizes.device

    return accumulate_sizes(sizes=token_sizes.to(device=device))


def head_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = head_catted_indices(token_sizes=sequence.token_sizes, device=sequence.data.device)

    return sequence.data[indices]


@torch.no_grad()
def last_catted_indices(token_sizes: Tensor, device: Device = None) -> Tensor:
    if device is None:
        device = token_sizes.device

    return torch.cumsum(token_sizes.to(device=device), dim=0) - 1


def last_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = last_catted_indices(token_sizes=sequence.token_sizes, device=sequence.data.device)

    return sequence.data[indices]


@torch.no_grad()
def init_catted_indices(token_sizes: Tensor, n: int = 1, device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes - n)

    return token_ptr + acc_token_sizes[batch_ptr]


def init_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices = init_catted_indices(token_sizes=sequence.token_sizes, n=n, device=sequence.data.device)

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes - n,
    )


@torch.no_grad()
def tail_catted_indices(token_sizes: Tensor, n: int = 1, device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes - n)

    return token_ptr + acc_token_sizes[batch_ptr] + n


def tail_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices = tail_catted_indices(token_sizes=sequence.token_sizes, n=n, device=sequence.data.device)

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes - n,
    )
