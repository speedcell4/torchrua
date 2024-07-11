from numbers import Number
from typing import Union

from torchrua.layout import C, L, P, R
from torchrua.utils import to_self


def to_cat(self: Union[L, P, R]) -> C:
    z = self.cat_view()
    return z._replace(data=self[z.ptr()])


C.cat = to_self
L.cat = to_cat
R.cat = to_cat
P.cat = to_cat


def cat_pack_to_left(self: C, fill_value: Number = 0) -> L:
    z = self.left_view(fill_value)
    z[self.ptr()] = self.data

    return z


def right_to_left(self: C, fill_value: Number = 0) -> L:
    z = self.left_view(fill_value)

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return z


C.left = cat_pack_to_left
L.left = to_self
P.left = cat_pack_to_left
R.left = right_to_left


def to_pack(self: C) -> P:
    z = self.pack_view()
    return z._replace(data=self[z.ptr()])


C.pack = to_pack
L.pack = to_pack
P.pack = to_self
R.pack = to_pack


def cat_pack_to_right(self: C, fill_value: Number = 0) -> R:
    z = self.right_view(fill_value)
    z[self.ptr()] = self.data

    return z


def left_to_right(self: C, fill_value: Number = 0) -> R:
    z = self.right_view(fill_value)

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return z


C.right = cat_pack_to_right
L.right = left_to_right
P.right = cat_pack_to_right
R.right = to_self
