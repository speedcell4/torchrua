from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number

from torchrua.core import CattedSequence
from torchrua.utils import sequence_shape

__all__ = [
    'empty_like', 'zeros_like', 'ones_like', 'full_like',
]


def empty_like(sequence: Union[CattedSequence, PackedSequence], batch_first: bool = True,
               dtype: torch.dtype = None, device: Device = None) -> Tensor:
    _, *sizes1 = sequence_shape(sequence, batch_first=batch_first)
    _, *sizes2 = sequence.data.size()

    return torch.empty(
        (*sizes1, *sizes2),
        dtype=dtype or sequence.data.dtype,
        device=device or sequence.data.device,
    )


def zeros_like(sequence: Union[CattedSequence, PackedSequence], batch_first: bool = True,
               dtype: torch.dtype = None, device: Device = None) -> Tensor:
    _, *sizes1 = sequence_shape(sequence, batch_first=batch_first)
    _, *sizes2 = sequence.data.size()

    return torch.zeros(
        (*sizes1, *sizes2),
        dtype=dtype or sequence.data.dtype,
        device=device or sequence.data.device,
    )


def ones_like(sequence: Union[CattedSequence, PackedSequence], batch_first: bool = True,
              dtype: torch.dtype = None, device: Device = None) -> Tensor:
    _, *sizes1 = sequence_shape(sequence, batch_first=batch_first)
    _, *sizes2 = sequence.data.size()

    return torch.ones(
        (*sizes1, *sizes2),
        dtype=dtype or sequence.data.dtype,
        device=device or sequence.data.device,
    )


def full_like(sequence: Union[CattedSequence, PackedSequence], fill_value: Number, batch_first: bool = True,
              dtype: torch.dtype = None, device: Device = None) -> Tensor:
    _, *sizes1 = sequence_shape(sequence, batch_first=batch_first)
    _, *sizes2 = sequence.data.size()

    return torch.full(
        (*sizes1, *sizes2), fill_value=fill_value,
        dtype=dtype or sequence.data.dtype,
        device=device or sequence.data.device,
    )
