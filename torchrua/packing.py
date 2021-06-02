from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.catting import cat_sequence, CattedSequence
from torchrua.indexing import lengths_to_ptr
from torchrua.utils import lengths_to_sorting_indices, get_device, accumulate_lengths

__all__ = [
    'pack_sequence',
    'pack_padded_sequence',
    'pack_catted_sequence',
]


def pack_sequence(sequences: List[Tensor]) -> PackedSequence:
    return pack_catted_sequence(sequence=cat_sequence(sequences=sequences))


def pack_padded_sequence(input: Tensor, lengths: Tensor,
                         batch_first: bool = False, enforce_sorted: bool = True) -> PackedSequence:
    device = get_device(input)

    if not enforce_sorted:
        sorted_indices, unsorted_indices = lengths_to_sorting_indices(lengths)
    else:
        sorted_indices = unsorted_indices = None

    batch_ptr, token_ptr, batch_sizes = lengths_to_ptr(
        lengths, sorted_indices=sorted_indices, device=device,
    )

    if batch_first:
        data = input[batch_ptr, token_ptr]
    else:
        data = input[token_ptr, batch_ptr]

    return PackedSequence(
        data=data,
        batch_sizes=batch_sizes.cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


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
