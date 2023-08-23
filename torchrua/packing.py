from typing import List

import torch
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import cat_sequence
from torchrua.core import _self
from torchrua.core import invert_permutation
from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T


def pack_sequence(sequence: List[T]) -> P:
    return cat_sequence(sequence=sequence).pack()


def pack_c(sequence: C) -> P:
    data, token_sizes = sequence
    b, t, *sizes = sequence.size()

    if len(sizes) > 0:
        return sequence.idx().pack().rua(sequence)

    _, sorting_indices = torch.sort(token_sizes.detach().cpu(), descending=True)
    unsorted_indices = invert_permutation(sorting_indices)

    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = unsorted_indices[batch_ptr]

    tensor = data.new_zeros((t, b))
    tensor[token_ptr, batch_ptr] = data

    mask = torch.zeros_like(tensor, dtype=torch.long)
    mask[token_ptr, batch_ptr] = 1

    return PackedSequence(
        data=tensor[mask.bool()],
        batch_sizes=mask.sum(dim=1).detach().cpu(),
        sorted_indices=sorting_indices,
        unsorted_indices=unsorted_indices,
    )


def pack_d(sequence: D) -> P:
    return sequence.idx().pack().rua(sequence)


C.pack = pack_c
D.pack = pack_d
P.pack = _self
