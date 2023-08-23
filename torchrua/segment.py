from typing import Union

import torch

from torchrua import PaddedSequence
from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P


def seg_c(sequence: C, duration: Union[C, D, P], fn) -> C:
    data, token_sizes = sequence
    duration = duration.cat()

    data = fn(data, duration.data)
    return duration._replace(data=data)


C.seg = seg_c


def seg_t(sequence: D, duration: Union[C, D, P], fn) -> D:
    duration, token_sizes = duration.pad(fill_value=0)

    remaining = sequence.size()[1] - duration.sum(dim=1, keepdim=True)
    duration = torch.cat([duration, remaining], dim=-1)

    data = fn(sequence._data(), duration.view(-1))
    data = data.view((*duration.size(), *data.size()[1:]))
    return PaddedSequence(data=data[:, :-1], token_sizes=token_sizes)


D.seg = seg_t
