from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, L, P, R


def cat_pack_to_left(self: C, fill_value: Number = 0) -> L:
    z = self.left_view(fill_value)
    z[self.ptr()] = self.data

    return z


C.left = cat_pack_to_left
P.left = cat_pack_to_left


def right_to_left(self: C, fill_value: Number = 0) -> L:
    z = self.left_view(fill_value)

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return z


L.left = to_self
R.left = right_to_left
