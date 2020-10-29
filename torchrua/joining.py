from typing import Optional, List, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

__all__ = [
    'cat_packed_sequences', 'cat_packed_data', 'cat_packed_batch_sizes',
    'stack_packed_sequences', 'stack_packed_data', 'stack_packed_batch_sizes',
]


def cat_packed_sequences(packs: List[PackedSequence]) -> PackedSequence:
    data = cat_packed_data(packs=packs)
    batch_sizes, sorted_indices, unsorted_indices = cat_packed_batch_sizes(pack=packs[0], num_packs=len(packs))

    return PackedSequence(
        data=data,
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def cat_packed_data(packs: List[PackedSequence]) -> Tensor:
    data = torch.stack([pack.data for pack in packs], dim=1)
    return rearrange(data, 'p n ... -> (p n) ...')


@torch.no_grad()
def cat_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.unsorted_indices is not None:
        unsorted_indices = torch.arange(num_packs, device=pack.data.device)
        unsorted_indices = unsorted_indices[:, None] + pack.unsorted_indices[None, :] * num_packs
        unsorted_indices = unsorted_indices.view(-1)

        sorted_indices = invert_permutation(unsorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    return pack.batch_sizes * num_packs, sorted_indices, unsorted_indices


def stack_packed_sequences(packs: List[PackedSequence]) -> PackedSequence:
    data = stack_packed_data(packs=packs)
    batch_sizes, sorted_indices, unsorted_indices = stack_packed_batch_sizes(pack=packs[0], num_packs=len(packs))

    return PackedSequence(
        data=data,
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def stack_packed_data(packs: List[PackedSequence]) -> Tensor:
    data = torch.stack([pack.data for pack in packs], dim=1)
    return rearrange(data, 'p n ... -> (p n) ...')


@torch.no_grad()
def stack_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.unsorted_indices is not None:
        unsorted_indices = torch.arange(num_packs, device=pack.data.device)
        unsorted_indices = unsorted_indices[None, :] + pack.unsorted_indices[:, None] * num_packs
        unsorted_indices = unsorted_indices.view(-1)

        sorted_indices = invert_permutation(unsorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    return pack.batch_sizes * num_packs, sorted_indices, unsorted_indices
