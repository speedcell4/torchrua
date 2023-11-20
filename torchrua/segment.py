from typing import Union

import torch

from torchrua.ty import C, D, P


def seg_c(sequence: C, duration: Union[C, D, P], fn) -> C:
    data, token_sizes = sequence
    duration = duration.cat()

    data = fn(data, duration.data)
    return duration._replace(data=data)


C.seg = seg_c


def seg_d(sequence: D, duration: Union[C, D, P], fn) -> D:
    duration, token_sizes = duration.pad(fill_value=0)

    remaining = sequence.size()[1] - duration.sum(dim=1, keepdim=True)
    duration = torch.cat([duration, remaining], dim=-1)

    data = fn(sequence._data(), duration.view(-1))
    data = data.view((*duration.size(), *data.size()[1:]))
    return D(data=data[:, :-1], token_sizes=token_sizes)


D.seg = seg_d


def seg_p(sequence: P, duration: Union[C, D, P], fn) -> P:
    return sequence.cat().seg(duration, fn).pack()


P.seg = seg_p
