from typing import List

import torch
from torch.types import Number

from torchrua.catting import cat_sequence
from torchrua.core import _self
from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import PaddedSequence
from torchrua.ty import T


def pad_sequence(sequence: List[T], fill_value: Number = 0) -> D:
    return cat_sequence(sequence).pad(fill_value=fill_value)


def pad_c(sequence: C, fill_value: Number = 0) -> D:
    data, token_sizes = sequence

    b, t, *sizes = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    return PaddedSequence(data=tensor, token_sizes=token_sizes)


def pad_p(sequence: P, fill_value: Number = 0) -> D:
    data, _, sorted_indices, _ = sequence

    b, t, *sizes = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    mask = data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return PaddedSequence(data=tensor, token_sizes=mask.sum(dim=1))


C.pad = pad_c
D.pad = _self
P.pad = pad_p
