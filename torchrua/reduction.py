from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.packing import pack_catted_sequence

__all__ = [
    'reduce_catted_sequence',
    'reduce_packed_sequence',
    'reduce_padded_sequence',
]


def reduce_catted_sequence(sequences: List[Tuple[Tensor, Tensor]], device: torch.device = None) -> PackedSequence:
    if device is None:
        device = sequences[0][0].device

    data, length1, length2 = zip(*[
        (sequence, lengths, lengths.size()[0])
        for sequence, lengths in sequences
    ])
    data = torch.cat(data, dim=0).to(device=device)
    length1 = torch.cat(length1, dim=0).to(device=device)
    length2 = torch.tensor(length2, dtype=torch.long, device=device)

    data_pack = pack_catted_sequence(sequence=data, lengths=length1)
    indices_pack = pack_catted_sequence(sequence=data_pack.unsorted_indices, lengths=length2)

    return PackedSequence(
        data=data_pack.data,
        batch_sizes=data_pack.batch_sizes,
        sorted_indices=invert_permutation(indices_pack.data),
        unsorted_indices=indices_pack.data,
    )


def reduce_packed_sequence(sequences: List[PackedSequence], device: torch.device = None) -> PackedSequence:
    raise NotImplementedError


def reduce_padded_sequence(sequence: Tensor, lengths: Tensor, device: torch.device = None) -> PackedSequence:
    raise NotImplementedError
