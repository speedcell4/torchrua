from typing import List

import torch
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.catting import CattedSequence
from torchrua.packing import pack_catted_sequence

__all__ = [
    'reduce_catted_sequence',
]


def reduce_catted_sequence(sequences: List[CattedSequence], device: torch.device = None) -> PackedSequence:
    if device is None:
        device = sequences[0].data.device

    data, length1, length2 = zip(*[
        (sequence.data, sequence.lengths, sequence.lengths.size()[0])
        for sequence in sequences
    ])
    data = torch.cat(data, dim=0).to(device=device)
    length1 = torch.cat(length1, dim=0).to(device=device)
    length2 = torch.tensor(length2, dtype=torch.long, device=device)

    data_pack = pack_catted_sequence(CattedSequence(data=data, lengths=length1))
    indices_pack = pack_catted_sequence(CattedSequence(data=data_pack.unsorted_indices, lengths=length2))

    return PackedSequence(
        data=data_pack.data,
        batch_sizes=data_pack.batch_sizes,
        sorted_indices=invert_permutation(indices_pack.data),
        unsorted_indices=indices_pack.data,
    )
