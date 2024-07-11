from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, L, P, R


def cat_pack_to_right(self: C, fill_value: Number = 0) -> R:
    z = self.right_view(fill_value)
    z[self.ptr()] = self.data

    return z


C.right = cat_pack_to_right
P.right = cat_pack_to_right


def left_to_right(self: C, fill_value: Number = 0) -> R:
    z = self.right_view(fill_value)

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return z


L.right = left_to_right
R.right = to_self
