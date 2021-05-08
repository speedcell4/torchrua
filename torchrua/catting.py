from typing import List, NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.utils import accumulate_lengths
from torchrua.indexing import lengths_to_ptr

__all__ = [
    'CattedSequence',
    'cat_sequence',
    'pack_catted_sequence',
]


class CattedSequence(NamedTuple):
    data: Tensor
    lengths: Tensor


def cat_sequence(sequences: List[Tensor]) -> CattedSequence:
    data, lengths = zip(*[
        (sequence, sequence.size()[0])
        for sequence in sequences
    ])

    data = torch.cat(data, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long, device=data.device)
    return CattedSequence(
        data=data,
        lengths=lengths,
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
