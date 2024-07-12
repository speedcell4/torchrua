import torch

from torchrua.layout import C, L, P, R, T, Z


def mask(self: Z, zero, one, dtype: torch.dtype = None) -> T:
    b, t, *_ = self.size()

    mask = self.data.new_full((b, t), fill_value=zero, dtype=dtype)
    mask[self.ptr()] = one

    return mask


C.mask = mask
L.mask = mask
P.mask = mask
R.mask = mask


def bmask(self: Z) -> T:
    return self.mask(zero=False, one=True, dtype=torch.bool)


C.bmask = bmask
L.bmask = bmask
P.bmask = bmask
R.bmask = bmask


def fmask(self: Z) -> T:
    return self.mask(zero=torch.finfo(self.data.dtype).min, one=0, dtype=self.data.dtype)


C.fmask = fmask
L.fmask = fmask
P.fmask = fmask
R.fmask = fmask
