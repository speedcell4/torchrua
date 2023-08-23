from typing import Union

import torch

from torchrua.ty import C
from torchrua.ty import D
from torchrua.ty import P
from torchrua.ty import T


def mask_cdp(sequence: Union[C, D, P]) -> T:
    b, t, *_ = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()

    mask = sequence.data.new_zeros((b, t), dtype=torch.bool)
    mask[batch_ptr, token_ptr] = True
    return mask


C.mask = mask_cdp
D.mask = mask_cdp
P.mask = mask_cdp
