from functools import singledispatch
from typing import Tuple, Union

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua import CattedSequence, major_sizes_to_size, major_sizes_to_ptr

__all__ = [
    'sequence_shape', 'catted_sequence_shape', 'packed_sequence_shape',
    'sequence_ptr', 'catted_sequence_ptr', 'packed_sequence_ptr',
]


@singledispatch
def sequence_shape(sequence: Union[CattedSequence, PackedSequence]) -> Tuple[int, int, int]:
    raise TypeError(f'type {type(sequence)} is not supported')


@sequence_shape.register
def catted_sequence_shape(sequence: CattedSequence) -> Tuple[int, int, int]:
    t, b = major_sizes_to_size(sizes=sequence.token_sizes)
    n, *_ = sequence.data.size()
    return n, b, t


@sequence_shape.register
def packed_sequence_shape(sequence: PackedSequence) -> Tuple[int, int, int]:
    b, t = major_sizes_to_size(sizes=sequence.batch_sizes)
    n, *_ = sequence.data.size()
    return n, b, t


@singledispatch
def sequence_ptr(sequence: Union[CattedSequence, PackedSequence], device: Device = None) -> Tuple[Tensor, Tensor]:
    raise TypeError(f'type {type(sequence)} is not supported')


@sequence_ptr.register
def catted_sequence_ptr(sequence: CattedSequence, device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequence.data.device

    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=sequence.token_sizes.to(device=device))
    return batch_ptr, token_ptr


@sequence_ptr.register
def packed_sequence_ptr(sequence: PackedSequence, device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequence.data.device

    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=sequence.batch_sizes.to(device=device))
    return batch_ptr, token_ptr
