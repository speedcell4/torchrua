import torch

from torchrua.core import get_offsets
from torchrua.ty import C, D, P, T


def head_c(sequence: C) -> T:
    data, token_sizes = sequence
    m, *_ = data.size()

    idx = get_offsets(sizes=token_sizes)
    return data[torch.clamp_max_(idx, max=m - 1)]


C.head = head_c


def head_d(sequence: D) -> T:
    data, _ = sequence
    return data[:, 0]


D.head = head_d


def head_p(sequence: P) -> T:
    data, batch_sizes, _, unsorted_indices = sequence
    if unsorted_indices is not None:
        return data[unsorted_indices]
    return data[:batch_sizes[0].detach().cpu().item()]


P.head = head_p
