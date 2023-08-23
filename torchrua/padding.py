from typing import List

import torch
from torch.types import Number

from torchrua.catting import cat_sequence
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

    data = torch.full(
        (b, t, *sizes), fill_value=fill_value,
        dtype=data.dtype, device=data.device,
    )
    data[batch_ptr, token_ptr] = sequence.data

    return PaddedSequence(data=data, token_sizes=token_sizes)


def pad_p(sequence: P, fill_value: Number = 0) -> D:
    data, _, sorted_indices, unsorted_indices = sequence

    b, t, *sizes = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    tensor = torch.full(
        (t, b, *sizes), fill_value=fill_value,
        dtype=data.dtype, device=data.device,
    )
    tensor[token_ptr, batch_ptr] = data

    mask = torch.zeros((t, b), dtype=torch.long, device=data.device)
    mask[token_ptr, batch_ptr] = 1

    return PaddedSequence(data=tensor.transpose(0, 1), token_sizes=mask.sum(dim=0))


C.pad = pad_c
D.pad = D.to
P.pad = pad_p
