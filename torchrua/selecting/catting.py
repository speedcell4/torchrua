from typing import Tuple

import torch
from torch import Tensor

from torchrua.catting import CattedSequence
from torchrua.core import batch_sizes_to_ptr
from torchrua.utils import accumulate_sizes

__all__ = [
    'head_catted_indices', 'head_catted_sequence',
    'last_catted_indices', 'last_catted_sequence',
    'init_catted_mask', 'init_catted_sequence',
    'tail_catted_mask', 'tail_catted_sequence',
    'reversed_catted_indices', 'reverse_catted_sequence',
    'rolled_catted_indices', 'roll_catted_sequence',
]


@torch.no_grad()
def head_catted_indices(sequence: CattedSequence) -> Tensor:
    return accumulate_sizes(sizes=sequence.token_sizes)


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


def head_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = head_catted_indices(sequence=sequence)
    return sequence.data[indices]


def last_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = last_catted_indices(sequence=sequence)
    return sequence.data[indices]


def init_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices, token_sizes = init_catted_mask(sequence=sequence, n=n)
    return CattedSequence(data=sequence.data[indices], token_sizes=token_sizes)


def tail_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices, token_sizes = tail_catted_mask(sequence=sequence, n=n)
    return CattedSequence(data=sequence.data[indices], token_sizes=token_sizes)


@torch.no_grad()
def reversed_catted_indices(sequence: CattedSequence) -> Tensor:
    token_sizes = sequence.token_sizes.to(device=sequence.data.device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    token_ptr = token_sizes[batch_ptr] - token_ptr - 1
    return token_ptr + acc_token_sizes[batch_ptr]


def reverse_catted_sequence(sequence: CattedSequence) -> CattedSequence:
    indices = reversed_catted_indices(sequence)
    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes,
    )


@torch.no_grad()
def rolled_catted_indices(sequence: CattedSequence, shifts: int) -> Tensor:
    token_sizes = sequence.token_sizes.to(device=sequence.data.device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    token_ptr = (token_ptr + token_sizes[batch_ptr] - shifts) % token_sizes[batch_ptr]
    return token_ptr + acc_token_sizes[batch_ptr]


def roll_catted_sequence(sequence: CattedSequence, shifts: int) -> CattedSequence:
    indices = rolled_catted_indices(sequence, shifts=shifts)
    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes,
    )
