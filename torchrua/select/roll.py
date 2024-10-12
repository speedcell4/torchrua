import torch

from torchrua.layout import C, L, P, R


def cat_roll(self: C, shifts: int) -> C:
    data, token_sizes = self

    batch_ptr, token_ptr = self.ptr()
    token_sizes = torch.repeat_interleave(token_sizes, token_sizes)
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return self._replace(data=self[batch_ptr, token_ptr])


C.roll = cat_roll


def left_roll(self: L, shifts: int) -> L:
    return self[self.idx().cat().roll(shifts).left()]


L.roll = left_roll


def pack_roll(self: P, shifts: int) -> P:
    return self[self.idx().cat().roll(shifts).pack()]


P.roll = pack_roll


def right_roll(self: R, shifts: int) -> R:
    return self[self.idx().cat().roll(shifts).right()]


R.roll = right_roll
