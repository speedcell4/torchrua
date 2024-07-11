import torch

from torchrua.layout import C, L, P, T


def last_c(sequence: C) -> T:
    data, token_sizes = sequence
    return data[token_sizes.cumsum(dim=0) - 1]


C.last = last_c


def last_d(sequence: L) -> T:
    b, t, *_ = sequence.size()
    batch_ptr = torch.arange(b, dtype=torch.long, device=sequence.data.device)
    return sequence.data[batch_ptr, sequence.token_sizes - 1]


L.last = last_d


def last_p(sequence: P) -> T:
    return sequence.idx().left().last().rua(sequence)


P.last = last_p
