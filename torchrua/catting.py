from typing import List

import torch

from torchrua.core import _self
from torchrua.ty import C, D, P, T


def cat_sequence(sequence: List[T]) -> C:
    data = torch.cat(sequence, dim=0)
    token_sizes = [s.size()[0] for s in sequence]
    return C(data=data, token_sizes=data.new_tensor(token_sizes, dtype=torch.long))


C.new = cat_sequence
C.cat = _self


def cat_t(sequence: T) -> C:
    token_sizes = sequence.new_tensor(sequence.size()[:1], dtype=torch.long)
    return C(data=sequence, token_sizes=token_sizes)


T.cat = cat_t


def cat_d(sequence: D) -> C:
    return sequence.idx().rua(sequence)


D.cat = cat_d


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

    return C(
        data=tensor[mask.bool()],
        token_sizes=mask.sum(dim=1),
    )


P.cat = cat_p
