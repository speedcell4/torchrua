from functools import singledispatch
from numbers import Number
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import cat_packed_sequence
from torchrua.catting import cat_padded_sequence
from torchrua.core import broadcast_devices
from torchrua.ty import CattedSequence


@singledispatch
def decode_sequence(sequence: Union[CattedSequence, PackedSequence, Tuple[Tensor, Tensor]]) -> List[List[Number]]:
    return decode_catted_sequence(cat_padded_sequence(*sequence))


@decode_sequence.register
def decode_catted_sequence(sequence: CattedSequence):
    sequence, token_sizes, _ = broadcast_devices(
        sequence.data.detach(), sequence.token_sizes.detach(),
        device=torch.device('cpu'),
    )

    return [sequence.tolist() for sequence in torch.split(sequence, token_sizes.tolist(), dim=0)]


@decode_sequence.register
def decode_packed_sequence(sequence: PackedSequence):
    return decode_catted_sequence(cat_packed_sequence(sequence))
