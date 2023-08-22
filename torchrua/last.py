from typing import Union

import torch

from torchrua.core import broadcast_devices
from torchrua.ty import C
from torchrua.ty import P
from torchrua.ty import T
from torchrua.ty import is_type

__all__ = [
    'last_sequence', 'last_catted_indices',
]


def last_sequence(sequence: Union[C, P]) -> T:
    if is_type(sequence, C):
        sequence, token_sizes = sequence
        indices = last_catted_indices(token_sizes, device=sequence.device)
        return sequence[indices]

    return sequence.data[sequence.idx().cat().last()]


C.last = last_sequence
P.last = last_sequence


def last_catted_indices(token_sizes: T, device: torch.device = None) -> T:
    token_sizes, _ = broadcast_devices(token_sizes, device=device)
    return token_sizes.cumsum(dim=0) - 1
