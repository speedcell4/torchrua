from typing import List
from typing import Union

import torch
from torch.types import Number

from torchrua.catting import cat_sequence
from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import Ts
from torchrua.ty import is_type


def split_sequence(sequence: Union[D, C, P]) -> Ts:
    if is_type(sequence, Union[D, P]):
        sequence = cat_sequence(sequence)

    sections = sequence.token_sizes.detach().cpu().tolist()
    return torch.split(sequence.data, sections, dim=0)


def decode_sequence(sequence: Union[D, C, P]) -> List[List[Number]]:
    return [tensor.detach().cpu().tolist() for tensor in split_sequence(sequence)]
