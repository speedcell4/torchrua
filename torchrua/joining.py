from typing import Optional, List, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import invert_permutation

from torchrua.indexing import lengths_to_ptr

__all__ = [
    'cat_packed_sequences', 'cat_packed_data', 'cat_packed_batch_sizes',
    'stack_packed_sequences', 'stack_packed_data', 'stack_packed_batch_sizes',
    'pack_catted_sequence',
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


def pack_catted_sequence(tensor: Tensor, lengths: Tensor) -> PackedSequence:
    sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
    unsorted_indices = invert_permutation(sorted_indices)

    batch_ptr, token_ptr, batch_sizes = lengths_to_ptr(
        lengths=sorted_lengths,
        sorted_indices=sorted_indices,
        device=sorted_lengths.device,
    )

    acc_lengths = F.pad(lengths.cumsum(dim=0), [1, -1])
    indices = acc_lengths[batch_ptr] + token_ptr

    return PackedSequence(
        data=tensor[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
