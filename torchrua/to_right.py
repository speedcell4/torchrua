import torch
from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, L, P, R


def cat_to_right(self: C, fill_value: Number = 0) -> R:
    data = self.data.new_full(self.size(), fill_value=fill_value)
    z = R(data=data, token_sizes=self.token_sizes)

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return z


C.right = cat_to_right
L.right = cat_to_right


def pack_to_right(self: P, fill_value: Number = 0) -> R:
    data, _, sorted_indices, _ = self

    b, t, *sizes = self.size()
    batch_ptr, token_ptr = self.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    mask = data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1
    token_sizes = mask.sum(dim=1)

    data = data.new_full((b, t, *sizes), fill_value=fill_value)
    z = R(data=data, token_sizes=token_sizes)
    z[batch_ptr, token_ptr] = self.data

    return z


P.right = pack_to_right

R.right = to_self
