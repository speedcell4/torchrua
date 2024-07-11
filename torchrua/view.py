from typing import Union

import torch

from torchrua import C, L, P, R, to_self
from torchrua.core import invert_permutation


def cat_view(self: P) -> C:
    b, t, *_ = self.size()
    batch_ptr, token_ptr = self.ptr()

    mask = self.data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return C(
        data=self.data,
        token_sizes=mask.sum(dim=1),
    )


C.cat_view = to_self
L.cat_view = cat_view
P.cat_view = cat_view
R.cat_view = cat_view


def pack_view(self: Union[C, L, R]) -> P:
    b, t, *_ = self.size()
    batch_ptr, token_ptr = self.ptr()

    mask = self.data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    _, index = torch.sort(self.token_sizes.detach().cpu(), descending=True)
    sorted_indices = index.to(device=self.data.device)

    unsorted_indices = invert_permutation(sorted_indices)

    return P(
        data=self.data,
        batch_sizes=mask.sum(dim=0).detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


C.pack_view = pack_view
L.pack_view = pack_view
P.pack_view = to_self
R.pack_view = pack_view
