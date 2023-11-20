from typing import List, Union

import torch
from torch.types import Number

from torchrua.ty import C, D, P, T


def split_cdp(sequence: Union[C, D, P]) -> List[T]:
    data, token_sizes = sequence.cat()

    return torch.split(data, token_sizes.cpu().tolist(), dim=0)


C.split = split_cdp
D.split = split_cdp
P.split = split_cdp


def tolist_cdp(sequence: Union[C, D, P]) -> List[List[Number]]:
    return [tensor.tolist() for tensor in sequence.detach().cpu().split()]


C.tolist = tolist_cdp
D.tolist = tolist_cdp
P.tolist = tolist_cdp
