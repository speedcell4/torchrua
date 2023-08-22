from typing import Union

import torch

from torchrua.ty import C
from torchrua.ty import P
from torchrua.ty import T


def mask_sequence(sequence: Union[C, P]) -> T:
    (b, t), (batch_ptr, token_ptr), token_sizes = sequence.ptr()

    mask = sequence.data.new_zeros((b, t), dtype=torch.bool)
    mask[batch_ptr, token_ptr] = True
    return mask


C.mask = mask_sequence
P.mask = mask_sequence
