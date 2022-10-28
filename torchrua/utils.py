from functools import singledispatch
from typing import Tuple, Union

from torch.nn.utils.rnn import PackedSequence

from torchrua import CattedSequence, major_sizes_to_size


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
