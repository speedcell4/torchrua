from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import major_sizes_to_ptr
from torchrua.utils import accumulate_sizes, resize_sizes, batch_sizes_to_token_sizes

__all__ = [
    'head_indices', 'select_head',
    'last_indices', 'select_last',
    'init_indices', 'select_init',
    'tail_indices', 'select_tail',
]


@torch.no_grad()
def head_indices(batch_sizes: Tensor, unsorted_indices: Optional[Tensor] = None) -> Tensor:
    if unsorted_indices is not None:
        return unsorted_indices

    return torch.arange(batch_sizes[0].item(), device=batch_sizes.device)


def select_head(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    device = sequence.data.device

    indices = head_indices(
        batch_sizes=sequence.batch_sizes.to(device=device),
        unsorted_indices=sequence.unsorted_indices if unsort else None,
    )
    return sequence.data[indices]


@torch.no_grad()
def last_indices(batch_sizes: Tensor, unsorted_indices: Optional[Tensor] = None) -> Tensor:
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr = head_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices)
    token_ptr = batch_sizes_to_token_sizes(batch_sizes=batch_sizes, batch_ptr=batch_ptr) - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_last(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    device = sequence.data.device

    indices = last_indices(
        batch_sizes=sequence.batch_sizes.to(device=device),
        unsorted_indices=sequence.unsorted_indices if unsort else None,
    )
    return sequence.data[indices]


@torch.no_grad()
def init_indices(batch_sizes: Tensor, n: int = 1) -> Tensor:
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_sizes = resize_sizes(sizes=batch_sizes, n=batch_sizes.size()[0] - n)
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_init(sequence: PackedSequence, n: int = 1) -> PackedSequence:
    device = sequence.data.device

    indices = init_indices(batch_sizes=sequence.batch_sizes.to(device=device), n=n)
    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes[n:],
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )


@torch.no_grad()
def tail_indices(batch_sizes: Tensor, n: int = 1) -> Tensor:
    return torch.arange(batch_sizes[0].item() * n, batch_sizes.sum().item(), device=batch_sizes.device)


def select_tail(sequence: PackedSequence, n: int = 1) -> PackedSequence:
    device = sequence.data.device

    indices = tail_indices(batch_sizes=sequence.batch_sizes.to(device=device), n=n)
    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes[n:],
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
