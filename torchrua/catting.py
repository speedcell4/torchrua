from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.indexing import batch_sizes_to_ptr, lengths_to_ptr
from torchrua.utils import accumulate_lengths, accumulate_batch_sizes

__all__ = [
    'cat_sequence',
    'cat_packed_sequence',
    'cat_padded_sequence',
]


def cat_sequence(sequences: List[Tensor], device: torch.device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequences[0].device

    data = torch.cat(sequences, dim=0)
    lengths = torch.tensor([
        sequence.size()[0] for sequence in sequences
    ], dtype=torch.long, device=device)
    return data, lengths


def cat_packed_sequence(sequence: PackedSequence, device: torch.device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequence[0].data.device

    batch_ptr, token_ptr, unsorted_lengths = batch_sizes_to_ptr(
        batch_sizes=sequence.batch_sizes,
        sorted_indices=None,
        unsorted_indices=sequence.unsorted_indices,
        total_length=None, device=device,
    )
    acc_batch_sizes = accumulate_batch_sizes(sequence.batch_sizes, device=device)
    acc_lengths = accumulate_lengths(unsorted_lengths)

    cat_index = acc_lengths[sequence.unsorted_indices[batch_ptr]] + token_ptr
    pack_index = acc_batch_sizes[token_ptr] + batch_ptr
    index = invert_permutation(cat_index)[pack_index]

    return sequence.data[index], unsorted_lengths


def cat_padded_sequence(sequence: Tensor, lengths: Tensor, batch_first: bool,
                        device: torch.device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequence[0].device

    cuda_unsorted_lengths = lengths.to(device=device)
    batch_ptr, token_ptr, batch_sizes = lengths_to_ptr(
        lengths=cuda_unsorted_lengths,
        sorted_indices=None,
        device=device,
    )
    acc_lengths = accumulate_lengths(cuda_unsorted_lengths)
    index = invert_permutation(acc_lengths[batch_ptr] + token_ptr)
    batch_ptr = batch_ptr[index]
    token_ptr = token_ptr[index]

    if batch_first:
        data = sequence[batch_ptr, token_ptr]
    else:
        data = sequence[token_ptr, batch_ptr]

    return data, cuda_unsorted_lengths
