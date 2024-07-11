import torch
from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, L, P, R


def cat_to_left(self: C, fill_value: Number = 0) -> L:
    z = L(
        data=self.data.new_full(self.size(), fill_value=fill_value),
        token_sizes=self.token_sizes,
    )

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return z


C.left = cat_to_left

L.left = to_self


def pack_to_left(self: P, fill_value: Number = 0) -> L:
    data, _, _, _ = self

    b, t, *sizes = self.size()
    batch_ptr, token_ptr = self.ptr()

    tensor = data.new_full((b, t, *sizes), fill_value=fill_value)
    tensor[batch_ptr, token_ptr] = data

    mask = data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return L(data=tensor, token_sizes=mask.sum(dim=1))


P.left = pack_to_left

R.left = cat_to_left
