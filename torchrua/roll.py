import torch

from torchrua.layout import C, L, P


def cat_roll(sequence: C, shifts: int) -> C:
    data, token_sizes = sequence

    batch_ptr, token_ptr = sequence.ptr()
    token_sizes = torch.repeat_interleave(token_sizes, token_sizes)
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return sequence._replace(data=sequence[batch_ptr, token_ptr])


C.roll = cat_roll


def left_roll(sequence: L, shifts: int) -> L:
    index1, _ = idx = sequence.idx()
    index2, _ = idx.roll(shifts)

    data = sequence.raw().clone()
    data[index1] = data[index2]

    return sequence._replace(data=data.view_as(sequence.data))


L.roll = left_roll


def pack_roll(sequence: P, shifts: int) -> P:
    return sequence[sequence.idx().cat().roll(shifts).pack()]


P.roll = pack_roll
