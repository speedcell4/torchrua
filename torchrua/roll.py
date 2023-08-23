import torch

from torchrua.ty import C
from torchrua.ty import CattedSequence
from torchrua.ty import P


def roll_c(sequence: C, shifts: int) -> C:
    data, token_sizes = sequence

    batch_ptr, token_ptr = sequence.ptr()
    token_sizes = torch.repeat_interleave(token_sizes, repeats=token_sizes)
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return CattedSequence(
        data=data[sequence.offsets()[batch_ptr] + token_ptr],
        token_sizes=sequence.token_sizes,
    )


def roll_p(sequence: P, shifts: int) -> P:
    return sequence.idx().cat().roll(shifts).pack().rua(sequence)


C.roll = roll_c
P.roll = roll_p
