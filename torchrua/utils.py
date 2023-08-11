from functools import singledispatch
from typing import Union

from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import CattedSequence
from torchrua.core import major_sizes_to_ptr
from torchrua.core import major_sizes_to_shape

__all__ = [
    'sequence_ptr', 'catted_sequence_ptr', 'packed_sequence_ptr',
    'sequence_shape', 'catted_sequence_shape', 'packed_sequence_shape',
]


@singledispatch
def sequence_ptr(sequence: Union[CattedSequence, PackedSequence], batch_first: bool = True, device: Device = None):
    raise TypeError(f'type {type(sequence)} is not supported')


@sequence_ptr.register
def catted_sequence_ptr(sequence: CattedSequence, batch_first: bool = True, device: Device = None):
    if device is None:
        device = sequence.data.device

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=sequence.token_sizes.to(device=device))
    return (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)


@sequence_ptr.register
def packed_sequence_ptr(sequence: PackedSequence, batch_first: bool = True, device: Device = None):
    if device is None:
        device = sequence.data.device

    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=sequence.batch_sizes.to(device=device))
    return (batch_ptr, token_ptr) if batch_first else (token_ptr, batch_ptr)


@singledispatch
def sequence_shape(sequence: Union[CattedSequence, PackedSequence], batch_first: bool = True):
    raise TypeError(f'type {type(sequence)} is not supported')


@sequence_shape.register
def catted_sequence_shape(sequence: CattedSequence, batch_first: bool = True):
    t, b = major_sizes_to_shape(sizes=sequence.token_sizes)
    n, *_ = sequence.data.size()
    return (n, b, t) if batch_first else (n, t, b)


@sequence_shape.register
def packed_sequence_shape(sequence: PackedSequence, batch_first: bool = True):
    b, t = major_sizes_to_shape(sizes=sequence.batch_sizes)
    n, *_ = sequence.data.size()
    return (n, b, t) if batch_first else (n, t, b)
