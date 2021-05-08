from typing import List, Tuple, Optional

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

__all__ = [
    'chunk_packed_sequence', 'chunk_packed_data', 'chunk_batch_sizes_dim0', 'chunk_batch_sizes_dim1',
]


def chunk_packed_sequence(pack: PackedSequence, chunks: int, dim: int) -> List[PackedSequence]:
    batch_sizes, sorted_indices, unsorted_indices = [
        chunk_batch_sizes_dim0,
        chunk_batch_sizes_dim1,
    ][dim](pack=pack, chunks=chunks)

    return [
        PackedSequence(
            data=data,
            batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )
        for data in chunk_packed_data(pack=pack, chunks=chunks)
    ]


def chunk_packed_data(pack: PackedSequence, chunks: int) -> List[Tensor]:
    data = rearrange(pack.data, '(p n) ... -> p n ...', n=chunks)
    return [data[:, index] for index in range(chunks)]


@torch.no_grad()
def chunk_batch_sizes_dim0(pack: PackedSequence, chunks: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.sorted_indices is not None:
        sorted_indices = pack.sorted_indices[::chunks]
    else:
        sorted_indices = None

    if pack.unsorted_indices is not None:
        unsorted_indices = pack.unsorted_indices[::chunks] // chunks
    else:
        unsorted_indices = None

    return pack.batch_sizes // chunks, sorted_indices, unsorted_indices


@torch.no_grad()
def chunk_batch_sizes_dim1(pack: PackedSequence, chunks: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.sorted_indices is not None:
        sorted_indices = pack.sorted_indices[::chunks]
    else:
        sorted_indices = None

    num_batches = pack.batch_sizes[0].item() // chunks
    if pack.unsorted_indices is not None:
        unsorted_indices = pack.unsorted_indices[:num_batches] // chunks
    else:
        unsorted_indices = None

    return pack.batch_sizes // chunks, sorted_indices, unsorted_indices
