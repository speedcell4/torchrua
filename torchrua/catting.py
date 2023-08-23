from typing import List

import torch

from torchrua.core import _self
from torchrua.ty import C
from torchrua.ty import CattedSequence
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T


def cat_sequence(sequence: List[T]) -> C:
    return CattedSequence(
        data=torch.cat(sequence, dim=0),
        token_sizes=torch.tensor([s.size()[0] for s in sequence], dtype=torch.long),
    )


def cat_d(sequence: D) -> C:
    return sequence.idx().rua(sequence)


def cat_p(sequence: P) -> C:
    data, batch_sizes, sorted_indices, unsorted_indices = sequence
    b, t, *sizes = sequence.size()

    if len(sizes) > 0:
        return sequence.idx().cat().rua(sequence)

    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    tensor = data.new_zeros((b, t))
    tensor[batch_ptr, token_ptr] = data

    mask = torch.zeros_like(tensor, dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return CattedSequence(
        data=tensor[mask.bool()],
        token_sizes=mask.sum(dim=1),
    )


C.cat = _self
D.cat = cat_d
P.cat = cat_p
