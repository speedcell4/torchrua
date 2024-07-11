import torch
from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, L, P, R


def cat_to_right(self: C, fill_value: Number = 0) -> R:
    data, token_sizes = self

    b, t, *sizes = self.size()
    batch_ptr, token_ptr = self.ptr()

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    return R(data=tensor, token_sizes=token_sizes)


C.right = cat_to_right


def left_to_right(self: L, fill_value: Number = 0) -> R:
    data, token_sizes = self

    b, t, *sizes = self.size()
    batch_ptr, token_ptr = self.ptr()

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    z = R(data=tensor, token_sizes=token_sizes)
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return R(data=tensor, token_sizes=token_sizes)


L.right = left_to_right


def pack_to_right(self: P, fill_value: Number = 0) -> R:
    data, _, sorted_indices, _ = self

    b, t, *sizes = self.size()
    batch_ptr, token_ptr = self.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    mask = data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return R(data=tensor, token_sizes=mask.sum(dim=1))


P.right = pack_to_right

R.right = to_self
