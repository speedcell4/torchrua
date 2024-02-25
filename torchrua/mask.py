from typing import Union

import torch
from torch.types import Number

from torchrua.ty import C, D, P, T


def mask_cd(sequence: Union[C, D, P], zero: Number = False, one: Number = True, dtype: torch.dtype = torch.bool) -> T:
    b, t, *_ = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()

    mask = sequence.data.new_full((b, t), fill_value=zero, dtype=dtype)
    mask[batch_ptr, token_ptr] = one
    return mask


C.mask = mask_cd
D.mask = mask_cd


def mask_p(sequence: Union[C, D, P], zero: Number = False, one: Number = True, dtype: torch.dtype = torch.bool) -> T:
    b, t, *_ = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()

    mask = sequence.data.new_full((b, t), fill_value=zero, dtype=dtype)
    mask[sequence.sorted_indices[batch_ptr], token_ptr] = one
    return mask


P.mask = mask_p
