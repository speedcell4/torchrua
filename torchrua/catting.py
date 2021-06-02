from typing import List, NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.indexing import batch_sizes_to_ptr
from torchrua.utils import accumulate_lengths, accumulate_batch_sizes

__all__ = [
    'CattedSequence',
    'cat_sequence',
    'cat_padded_sequence',
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


def cat_packed_sequence(sequence: PackedSequence) -> CattedSequence:
    device = sequence[0].data.device

    batch_ptr, token_ptr, lengths = batch_sizes_to_ptr(
        batch_sizes=sequence.batch_sizes,
        sorted_indices=None,
        unsorted_indices=sequence.unsorted_indices,
        total_length=None, device=device,
    )
    acc_batch_sizes = accumulate_batch_sizes(sequence.batch_sizes, device=device)
    acc_lengths = accumulate_lengths(lengths)

    cat_index = acc_lengths[sequence.unsorted_indices[batch_ptr]] + token_ptr
    pack_index = acc_batch_sizes[token_ptr] + batch_ptr
    index = invert_permutation(cat_index)[pack_index]

    return CattedSequence(data=sequence.data[index], lengths=lengths)


def cat_padded_sequence(sequence: Tensor, lengths: Tensor, batch_first: bool) -> CattedSequence:
    raise NotImplementedError
