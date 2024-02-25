import torch

from torchrua.ty import C, D, P


def roll_c(sequence: C, shifts: int) -> C:
    data, token_sizes = sequence

    batch_ptr, token_ptr = sequence.ptr()
    token_sizes = torch.repeat_interleave(token_sizes, token_sizes)
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return sequence[batch_ptr, token_ptr]


C.roll = roll_c


def roll_d(sequence: D, shifts: int) -> D:
    index1, _ = idx = sequence.idx()
    index2, _ = idx.roll(shifts)

    data = sequence._data().clone()
    data[index1] = data[index2]

    return sequence._replace(data=data.view_as(sequence.data))


D.roll = roll_d


def roll_p(sequence: P, shifts: int) -> P:
    return sequence.idx().cat().roll(shifts).pack().rua(sequence)


P.roll = roll_p
