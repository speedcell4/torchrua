from typing import List

import torch

from torchrua.ty import C
from torchrua.ty import CattedSequence
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T

__all__ = [
    'cat_sequence',
]


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

    mask = data.new_zeros((t, b), dtype=torch.long)
    mask[token_ptr, batch_ptr] = 1

    tensor = torch.zeros_like(mask, dtype=data.dtype)
    tensor[token_ptr, batch_ptr] = data

    return CattedSequence(
        data=tensor.t()[mask.t().bool()],
        token_sizes=mask.sum(dim=0),
    )


C.cat = C.to
D.cat = cat_d
P.cat = cat_p
