import torch

from torchrua.core import major_sizes_to_size
from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T


def last_c(sequence: C) -> T:
    data, token_sizes = sequence
    return data[token_sizes.cumsum(dim=0) - 1]


C.last = last_c


def last_d(sequence: D) -> T:
    t, b = major_sizes_to_size(sequence.token_sizes)
    batch_ptr = torch.arange(b, dtype=torch.long, device=sequence.data.device)
    return sequence.data[batch_ptr, sequence.token_sizes - 1]


D.last = last_d


def last_p(sequence: P) -> T:
    return sequence.idx().cat().last().rua(sequence)


P.last = last_p
