import torch

from torchrua.ty import C, D, P


def rev_c(sequence: C) -> C:
    data, token_sizes = sequence
    _, _, *sizes = sequence.size()

    if len(sizes) > 0:
        return sequence.idx().rev().rua(sequence)

    data = torch.flip(data, dims=[0])
    token_sizes = torch.flip(token_sizes, dims=[0])

    data = torch.split(data, token_sizes.detach().cpu().tolist(), dim=0)
    return sequence._replace(data=torch.cat(data[::-1], dim=0))


C.rev = rev_c


def rev_d(sequence: D) -> D:
    index1, _ = idx = sequence.idx()
    index2, _ = idx.rev()

    data = sequence._data().clone()
    data[index1] = data[index2]

    return sequence._replace(data=data.view_as(sequence.data))


D.rev = rev_d


def rev_p(sequence: P) -> P:
    return sequence.idx().cat().rev().pack().rua(sequence)


P.rev = rev_p
