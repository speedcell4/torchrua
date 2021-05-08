from typing import Optional, List, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import invert_permutation

from torchrua.catting import CattedSequence
from torchrua.indexing import lengths_to_ptr
from torchrua.utils import accumulate_lengths

__all__ = [
    'stack_packed_sequences', 'stack_data', 'stack_indices_dim0', 'stack_indices_dim1',
    'pack_catted_sequence',
]


def stack_packed_sequences(packs: List[PackedSequence], dim: int) -> PackedSequence:
    batch_sizes, sorted_indices, unsorted_indices = [
        stack_indices_dim0, stack_indices_dim1,
    ][dim](pack=packs[0], num_packs=len(packs))

    return PackedSequence(
        data=stack_data(packs=packs),
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def stack_data(packs: List[PackedSequence]) -> Tensor:
    data = torch.stack([pack.data for pack in packs], dim=1)
    return rearrange(data, 'p n ... -> (p n) ...')


@torch.no_grad()
def stack_indices_dim0(pack: PackedSequence, num_packs: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.unsorted_indices is not None:
        unsorted_indices = torch.arange(num_packs, device=pack.data.device)
        unsorted_indices = unsorted_indices[None, :] + pack.unsorted_indices[:, None] * num_packs
        unsorted_indices = unsorted_indices.view(-1)

        sorted_indices = invert_permutation(unsorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    return pack.batch_sizes * num_packs, sorted_indices, unsorted_indices


@torch.no_grad()
def stack_indices_dim1(pack: PackedSequence, num_packs: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.unsorted_indices is not None:
        unsorted_indices = torch.arange(num_packs, device=pack.data.device)
        unsorted_indices = unsorted_indices[:, None] + pack.unsorted_indices[None, :] * num_packs
        unsorted_indices = unsorted_indices.view(-1)

        sorted_indices = invert_permutation(unsorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    return pack.batch_sizes * num_packs, sorted_indices, unsorted_indices


def pack_catted_sequence(sequence: CattedSequence) -> PackedSequence:
    sorted_lengths, sorted_indices = torch.sort(sequence.lengths, descending=True)
    unsorted_indices = invert_permutation(sorted_indices)

    batch_ptr, token_ptr, batch_sizes = lengths_to_ptr(
        lengths=sorted_lengths,
        sorted_indices=sorted_indices,
        device=sorted_lengths.device,
    )

    acc_lengths = accumulate_lengths(lengths=sequence.lengths)
    indices = acc_lengths[batch_ptr] + token_ptr

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
