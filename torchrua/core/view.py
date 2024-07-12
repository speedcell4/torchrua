from numbers import Number
from typing import Union

import torch
from torch import Tensor

from torchrua.layout import C, L, P, R, Z
from torchrua.utils import invert_permutation, to_self


def get_mask(self: Z) -> Tensor:
    b, t, *_ = self.size()
    batch_ptr, token_ptr = self.ptr()

    mask = self.data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return mask


def cat_view(self: Union[L, P, R], **kwargs) -> C:
    return C(
        data=self.data,
        token_sizes=get_mask(self).sum(dim=1),
    )


C.cat_view = to_self
L.cat_view = cat_view
P.cat_view = cat_view
R.cat_view = cat_view


def left_view(self: Union[C, P, R], fill_value: Number, dtype: torch.dtype = None) -> L:
    return L(
        data=self.data.new_full(self.size(), fill_value=fill_value, dtype=dtype),
        token_sizes=get_mask(self).sum(dim=1),
    )


C.left_view = left_view
L.left_view = to_self
P.left_view = left_view
R.left_view = left_view


def pack_view(self: Union[C, L, R], **kwargs) -> P:
    _, index = torch.sort(self.token_sizes.detach().cpu(), descending=True)
    sorted_indices = index.to(device=self.data.device)

    unsorted_indices = invert_permutation(sorted_indices)

    return P(
        data=self.data,
        batch_sizes=get_mask(self).sum(dim=0).detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


C.pack_view = pack_view
L.pack_view = pack_view
P.pack_view = to_self
R.pack_view = pack_view


def right_view(self: Union[C, L, P], fill_value: Number, dtype: torch.dtype = None) -> R:
    return R(
        data=self.data.new_full(self.size(), fill_value=fill_value, dtype=dtype),
        token_sizes=get_mask(self).sum(dim=1),
    )


C.right_view = right_view
L.right_view = right_view
P.right_view = right_view
R.right_view = to_self
