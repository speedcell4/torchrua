from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

__all__ = [
    'chunk_packed_sequence', 'chunk_data', 'chunk_batch_sizes_dim0', 'chunk_batch_sizes_dim1',
]


def chunk_packed_sequence(sequence: PackedSequence, chunks: int, dim: int) -> List[PackedSequence]:
    batch_sizes, sorted_indices, unsorted_indices = [
        chunk_batch_sizes_dim0,
        chunk_batch_sizes_dim1,
    ][dim](sequence=sequence, chunks=chunks)

    return [
        PackedSequence(
            data=data,
            batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )
        for data in chunk_data(sequence=sequence, chunks=chunks)
    ]


def chunk_data(sequence: PackedSequence, chunks: int) -> List[Tensor]:
    return [sequence.data[index::chunks] for index in range(chunks)]


@torch.no_grad()
def chunk_batch_sizes_dim0(sequence: PackedSequence, chunks: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if sequence.sorted_indices is not None:
        sorted_indices = sequence.sorted_indices[::chunks]
    else:
        sorted_indices = None

    if sequence.unsorted_indices is not None:
        unsorted_indices = sequence.unsorted_indices[::chunks] // chunks
    else:
        unsorted_indices = None

    return sequence.batch_sizes // chunks, sorted_indices, unsorted_indices


@torch.no_grad()
def chunk_batch_sizes_dim1(sequence: PackedSequence, chunks: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if sequence.sorted_indices is not None:
        sorted_indices = sequence.sorted_indices[::chunks]
    else:
        sorted_indices = None

    num_batches = sequence.batch_sizes[0].item() // chunks
    if sequence.unsorted_indices is not None:
        unsorted_indices = sequence.unsorted_indices[:num_batches] // chunks
    else:
        unsorted_indices = None

    return sequence.batch_sizes // chunks, sorted_indices, unsorted_indices
