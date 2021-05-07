from typing import List, NamedTuple

import torch
from torch import Tensor


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
