from torch.types import Number

from torchrua import to_self
from torchrua.layout import C, L, P, R


def cat_pack_to_left(self: C, fill_value: Number = 0) -> L:
    z = L(
        data=self.data.new_full(self.size(), fill_value=fill_value),
        token_sizes=self.cat_view().token_sizes,
    )

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self.data

    return z


C.left = cat_pack_to_left
P.left = cat_pack_to_left


def right_to_left(self: C, fill_value: Number = 0) -> L:
    z = L(
        data=self.data.new_full(self.size(), fill_value=fill_value),
        token_sizes=self.cat_view().token_sizes,
    )

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self[batch_ptr, token_ptr]

    return z


L.left = to_self
R.left = right_to_left
