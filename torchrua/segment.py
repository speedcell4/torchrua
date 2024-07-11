from typing import Union

import torch

from torchrua.layout import C, L, P


def seg_c(sequence: C, duration: Union[C, L, P], fn) -> C:
    data, token_sizes = sequence
    duration = duration.cat()

    data = fn(data, duration.data)
    return duration._replace(data=data)


C.seg = seg_c


def seg_d(sequence: L, duration: Union[C, L, P], fn) -> L:
    duration, token_sizes = duration.pad(fill_value=0)

    remaining = sequence.size()[1] - duration.sum(dim=1, keepdim=True)
    duration = torch.cat([duration, remaining], dim=-1)

    data = fn(sequence.raw(), duration.view(-1))
    data = data.view((*duration.size(), *data.size()[1:]))
    return L(data=data[:, :-1], token_sizes=token_sizes)


L.seg = seg_d


def seg_p(sequence: P, duration: Union[C, L, P], fn) -> P:
    return sequence.cat().seg(duration, fn).pack()


P.seg = seg_p
