from typing import Optional, List, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import invert_permutation

__all__ = [
    'stack_packed_sequences', 'stack_data',
    'stack_indices_dim0', 'stack_indices_dim1',
]


def stack_packed_sequences(sequences: List[PackedSequence], dim: int) -> PackedSequence:
    batch_sizes, sorted_indices, unsorted_indices = [
        stack_indices_dim0, stack_indices_dim1,
    ][dim](sequence=sequences[0], chunks=len(sequences))

    return PackedSequence(
        data=stack_data(sequences=sequences),
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def stack_data(sequences: List[PackedSequence]) -> Tensor:
    data = torch.stack([sequence.data for sequence in sequences], dim=1)
    return rearrange(data, 'p n ... -> (p n) ...')


@torch.no_grad()
def stack_indices_dim0(sequence: PackedSequence, chunks: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if sequence.unsorted_indices is not None:
        unsorted_indices = torch.arange(chunks, device=sequence.data.device)
        unsorted_indices = unsorted_indices[None, :] + sequence.unsorted_indices[:, None] * chunks
        unsorted_indices = unsorted_indices.view(-1)

        sorted_indices = invert_permutation(unsorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    return sequence.batch_sizes * chunks, sorted_indices, unsorted_indices


@torch.no_grad()
def stack_indices_dim1(sequence: PackedSequence, chunks: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if sequence.unsorted_indices is not None:
        unsorted_indices = torch.arange(chunks, device=sequence.data.device)
        unsorted_indices = unsorted_indices[:, None] + sequence.unsorted_indices[None, :] * chunks
        unsorted_indices = unsorted_indices.view(-1)

        sorted_indices = invert_permutation(unsorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    return sequence.batch_sizes * chunks, sorted_indices, unsorted_indices
