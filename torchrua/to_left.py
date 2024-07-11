import torch
from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, L, P, T

L.left = to_self


def left_l(sequence: T, fill_value: Number = 0) -> L:
    token_sizes = sequence.new_tensor(sequence.size()[:1], dtype=torch.long)
    return L(data=sequence[None], token_sizes=token_sizes)


T.left = left_l


def left_c(sequence: C, fill_value: Number = 0) -> L:
    data, token_sizes = sequence

    b, t, *sizes = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    return L(data=tensor, token_sizes=token_sizes)


C.left = left_c


def left_p(sequence: P, fill_value: Number = 0) -> L:
    data, _, sorted_indices, _ = sequence

    b, t, *sizes = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    mask = data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return L(data=tensor, token_sizes=mask.sum(dim=1))


P.left = left_p
