from typing import Union

import torch

from torchrua.layout import C, L, P, R
from torchrua.utils import to_self

C.cat = to_self


def left_right_to_cat(self: Union[L, R]) -> C:
    return self[self.idx()]


L.cat = left_right_to_cat
R.cat = left_right_to_cat


def pack_to_cat(self: P) -> C:
    z = C(
        data=torch.empty_like(self.data),
        token_sizes=self.token_sizes,
    )

    batch_ptr, token_ptr = self.ptr()
    z[batch_ptr, token_ptr] = self.data

    return z


P.cat = pack_to_cat
