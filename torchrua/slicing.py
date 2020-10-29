from typing import List, Tuple, Optional

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

__all__ = [
    'uncat_packed_sequence', 'uncat_packed_data', 'uncat_packed_batch_sizes',
    'unstack_packed_sequence', 'unstack_packed_data', 'unstack_packed_batch_sizes',
]


def uncat_packed_sequence(pack: PackedSequence, num_packs: int) -> List[PackedSequence]:
    batch_sizes, sorted_indices, unsorted_indices = uncat_packed_batch_sizes(pack=pack, num_packs=num_packs)
    return [
        PackedSequence(
            data=data, batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )
        for data in uncat_packed_data(pack=pack, num_packs=num_packs)
    ]


def uncat_packed_data(pack: PackedSequence, num_packs: int) -> List[Tensor]:
    data = rearrange(pack.data, '(p n) ... -> p n ...', n=num_packs)
    return [data[:, index] for index in range(num_packs)]


@torch.no_grad()
def uncat_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> \
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.sorted_indices is not None:
        sorted_indices = pack.sorted_indices[::num_packs]
    else:
        sorted_indices = None

    num_batches = pack.batch_sizes[0].item() // num_packs
    if pack.unsorted_indices is not None:
        unsorted_indices = pack.unsorted_indices[:num_batches] // num_packs
    else:
        unsorted_indices = None

    return pack.batch_sizes // num_packs, sorted_indices, unsorted_indices


def unstack_packed_sequence(pack: PackedSequence, num_packs: int) -> List[PackedSequence]:
    batch_sizes, sorted_indices, unsorted_indices = unstack_packed_batch_sizes(pack=pack, num_packs=num_packs)
    return [
        PackedSequence(
            data=data, batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )
        for data in unstack_packed_data(pack=pack, num_packs=num_packs)
    ]


def unstack_packed_data(pack: PackedSequence, num_packs: int) -> List[Tensor]:
    data = rearrange(pack.data, '(p n) ... -> p n ...', n=num_packs)
    return [data[:, index] for index in range(num_packs)]


@torch.no_grad()
def unstack_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> \
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.sorted_indices is not None:
        sorted_indices = pack.sorted_indices[::num_packs]
    else:
        sorted_indices = None

    if pack.unsorted_indices is not None:
        unsorted_indices = pack.unsorted_indices[::num_packs] // num_packs
    else:
        unsorted_indices = None

    return pack.batch_sizes // num_packs, sorted_indices, unsorted_indices
