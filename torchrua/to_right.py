import torch
from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, P, R, T

R.right = to_self


def right_l(sequence: T, fill_value: Number = 0) -> R:
    token_sizes = sequence.new_tensor(sequence.size()[:1], dtype=torch.long)
    return R(data=sequence[None], token_sizes=token_sizes)


T.right = right_l


def right_c(sequence: C, fill_value: Number = 0) -> R:
    data, token_sizes = sequence

    b, t, *sizes = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    return R(data=tensor, token_sizes=token_sizes)


C.right = right_c


def right_p(sequence: P, fill_value: Number = 0) -> R:
    data, _, sorted_indices, _ = sequence

    b, t, *sizes = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    mask = data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return R(data=tensor, token_sizes=mask.sum(dim=1))


P.right = right_p
