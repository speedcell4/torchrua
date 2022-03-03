from typing import Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_ptr

__all__ = [
    'unique_catted_sequence',
    'unique_packed_sequence',
]


def unique_catted_sequence(sequence: CattedSequence, device: Device = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = sequence.data.device

    data = sequence.data.to(device=device)
    token_sizes = sequence.token_sizes.to(device=device)

    unique1, token_ptr = torch.unique(data, sorted=True, return_inverse=True)
    batch_ptr = torch.repeat_interleave(token_sizes)

    n = unique1.size()[0]
    unique2, inverse_ptr, counts = torch.unique(
        n * batch_ptr + token_ptr,
        sorted=True, return_inverse=True, return_counts=True,
    )

    return unique1[unique2 % n], inverse_ptr, counts


def unique_packed_sequence(sequence: PackedSequence, device: Device = None) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device = sequence.data.device

    data = sequence.data.to(device=device)
    batch_sizes = sequence.batch_sizes.to(device=device)

    unique1, token_ptr = torch.unique(data, sorted=True, return_inverse=True)
    batch_ptr, _ = major_sizes_to_ptr(sizes=batch_sizes)

    n = unique1.size()[0]
    unique2, inverse_ptr, counts = torch.unique(
        n * batch_ptr + token_ptr,
        sorted=True, return_inverse=True, return_counts=True,
    )

    return unique1[unique2 % n], inverse_ptr, counts
