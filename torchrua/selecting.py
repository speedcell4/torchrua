from typing import Tuple

import torch
from torch import Tensor

from torchrua import CattedSequence
from torchrua.catting import cat_sequence
from torchrua.indexing import batch_sizes_to_ptr
from torchrua.utils import accumulate_sizes


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

    return token_sizes[batch_ptr] - token_ptr - 1 + acc_token_sizes[batch_ptr]


def reverse_catted_sequence(sequence: CattedSequence) -> CattedSequence:
    indices = reversed_catted_indices(sequence)
    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes,
    )


@torch.no_grad()
def rolled_catted_indices(sequence: CattedSequence, shifts: int) -> Tensor:
    raise NotImplementedError


def roll_catted_sequence(sequence: CattedSequence, shifts: int) -> CattedSequence:
    raise NotImplementedError


if __name__ == '__main__':
    data = cat_sequence([
        torch.arange(5),
        torch.arange(2),
        torch.arange(3),
    ])
    print(data)
    print(reverse_catted_sequence(data))
