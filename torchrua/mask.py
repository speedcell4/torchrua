from typing import Union

import torch
from torch.types import Number

from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T


def mask_cdp(sequence: Union[C, D, P], zero: Number = False, one: Number = True, dtype: torch.dtype = torch.bool) -> T:
    b, t, *_ = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()

    mask = sequence.data.new_full((b, t), fill_value=zero, dtype=dtype)
    mask[batch_ptr, token_ptr] = one
    return mask


C.mask = mask_cdp
D.mask = mask_cdp
P.mask = mask_cdp
