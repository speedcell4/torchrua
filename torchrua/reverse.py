from torchrua.ty import C
from torchrua.ty import CattedSequence
from torchrua.ty import P


def rev_c(sequence: C) -> C:
    data, token_sizes = sequence

    batch_ptr, token_ptr = sequence.ptr()
    token_ptr = (token_sizes - 1)[batch_ptr] - token_ptr

    return CattedSequence(
        data=data[sequence.offsets()[batch_ptr] + token_ptr],
        token_sizes=token_sizes,
    )


def rev_p(sequence: P) -> P:
    return sequence.idx().cat().rev().pack().rua(sequence)


C.rev = rev_c
P.rev = rev_p
