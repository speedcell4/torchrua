from typing import Tuple

import torch
from torch import Tensor
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import batch_sizes_to_ptr
from torchrua.utils import accumulate_sizes

__all__ = [
    'head_catted_indices', 'head_catted_sequence',
    'last_catted_indices', 'last_catted_sequence',
    'init_catted_mask', 'init_catted_sequence',
    'tail_catted_mask', 'tail_catted_sequence',
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
def last_catted_indices(sequence: CattedSequence) -> Tensor:
    return torch.cumsum(sequence.token_sizes, dim=0) - 1


@torch.no_grad()
def init_catted_mask(sequence: CattedSequence, n: int = 1) -> Tuple[Tensor, Tensor]:
    assert (sequence.token_sizes >= n).all().item()

    token_sizes = sequence.token_sizes.to(device=sequence.data.device)
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    token_sizes = token_sizes - n
    return token_ptr < token_sizes[batch_ptr], token_sizes


@torch.no_grad()
def tail_catted_mask(sequence: CattedSequence, n: int = 1) -> Tuple[Tensor, Tensor]:
    assert (sequence.token_sizes >= n).all().item()

    token_sizes = sequence.token_sizes.to(device=sequence.data.device)
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    return token_ptr >= n, token_sizes - n


def last_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = last_catted_indices(sequence=sequence)
    return sequence.data[indices]


def init_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices, token_sizes = init_catted_mask(sequence=sequence, n=n)
    return CattedSequence(data=sequence.data[indices], token_sizes=token_sizes)


def tail_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices, token_sizes = tail_catted_mask(sequence=sequence, n=n)
    return CattedSequence(data=sequence.data[indices], token_sizes=token_sizes)
