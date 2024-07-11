import torch

from torchrua.layout import C, L, P, R
from torchrua.utils import to_self

C.cat = to_self


def left_to_cat(self: L) -> C:
    return self[self.idx()]


L.cat = left_to_cat


def pack_to_cat(self: P) -> C:
    b, t, *sizes = self.size()

    if len(sizes) > 0:
        return self[self.idx().cat()]

    data, batch_sizes, sorted_indices, _ = self
    batch_ptr, token_ptr = self.ptr()

    tensor = data.new_zeros((b, t))
    tensor[batch_ptr, token_ptr] = data

    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask[batch_ptr, token_ptr] = True

    return C(
        data=tensor[mask],
        token_sizes=mask.long().sum(dim=1),
    )


P.cat = pack_to_cat


def right_to_cat(self: R) -> C:
    return self[self.idx()]


R.cat = right_to_cat
