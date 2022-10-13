from functools import singledispatch
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_size, major_sizes_to_ptr

Sequence = Union[CattedSequence, PackedSequence]


@singledispatch
def size(sequence, batch_first: bool = True) -> Tuple[int, ...]:
    raise TypeError(f'type {type(sequence)} is not supported')


@size.register
def packed_size(sequence: PackedSequence, batch_first: bool = True) -> Tuple[int, ...]:
    b, t = major_sizes_to_size(sizes=sequence.batch_sizes)
    return *((b, t) if batch_first else (t, b)), *sequence.data.size()[1:]


@size.register
def catted_size(sequence: CattedSequence, batch_first: bool = True) -> Tuple[int, ...]:
    t, b = major_sizes_to_size(sizes=sequence.token_sizes)
    return *((b, t) if batch_first else (t, b)), *sequence.data.size()[1:]


@singledispatch
def ptr(sequence, batch_first: bool = True) -> Tuple[Tensor, Tensor]:
    raise TypeError(f'type {type(sequence)} is not supported')


@ptr.register
def packed_ptr(sequence: PackedSequence, batch_first: bool = True) -> Tuple[Tensor, Tensor]:
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=sequence.batch_sizes)
    return (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)


@ptr.register
def catted_ptr(sequence: CattedSequence, batch_first: bool = True) -> Tuple[Tensor, Tensor]:
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=sequence.token_sizes)
    return (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)


def empty_like(sequence: Sequence, batch_first: bool = True,
               dtype: torch.dtype = None, device: Device = None,
               requires_grad: bool = False, *args, **kwargs) -> Tensor:
    return torch.empty(
        size=size(sequence, batch_first=batch_first),
        dtype=dtype, device=device, requires_grad=requires_grad, *args, **kwargs,
    )


def zeros_like(sequence: Sequence, batch_first: bool = True,
               dtype: torch.dtype = None, device: Device = None,
               requires_grad: bool = False, *args, **kwargs) -> Tensor:
    return torch.zeros(
        size=size(sequence, batch_first=batch_first),
        dtype=dtype, device=device, requires_grad=requires_grad, *args, **kwargs,
    )


def ones_like(sequence: Sequence, batch_first: bool = True,
              dtype: torch.dtype = None, device: Device = None,
              requires_grad: bool = False, *args, **kwargs) -> Tensor:
    return torch.ones(
        size=size(sequence, batch_first=batch_first),
        dtype=dtype, device=device, requires_grad=requires_grad, *args, **kwargs,
    )


def full_like(sequence: Sequence, fill_value: Number, batch_first: bool = True,
              dtype: torch.dtype = None, device: Device = None,
              requires_grad: bool = False, *args, **kwargs) -> Tensor:
    return torch.full(
        size=size(sequence, batch_first=batch_first), fill_value=fill_value,
        dtype=dtype, device=device, requires_grad=requires_grad, *args, **kwargs,
    )
