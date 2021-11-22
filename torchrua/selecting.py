from typing import Tuple

import torch
from torch import Tensor
from torchrua.indexing import token_sizes_to_ptr, batch_sizes_to_ptr

from torchrua.catting import cat_sequence

from torchrua import CattedSequence
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
def tail_catted_indices(sequence: CattedSequence, drop_n: int = 1) -> Tensor:
    raise NotImplementedError


def head_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = head_catted_indices(sequence=sequence)
    return sequence.data[indices]


def last_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = last_catted_indices(sequence=sequence)
    return sequence.data[indices]


def init_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices, token_sizes = init_catted_mask(sequence=sequence, n=n)
    return CattedSequence(data=sequence.data[indices], token_sizes=token_sizes)


def tail_catted_sequence(sequence: CattedSequence, drop_n: int = 1) -> CattedSequence:
    raise NotImplementedError


if __name__ == '__main__':
    data = cat_sequence([
        torch.arange(5),
        torch.arange(2),
        torch.arange(3),
    ])
    print(data)
    print(init_catted_sequence(data, n=2))
