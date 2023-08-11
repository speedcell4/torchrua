from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import cat_sequence
from torchrua.core import CattedSequence
from torchrua.core import accumulate_sizes
from torchrua.core import broadcast_devices
from torchrua.core import get_device
from torchrua.core import sizes_to_sorting
from torchrua.info import token_sizes_to_minor_ptr3

__all__ = [
    'pack_sequence',
    'pack_catted_indices', 'pack_catted_sequence',
    'pack_padded_indices', 'pack_padded_sequence',
]


def pack_sequence(sequences: List[Tensor], device: torch.device = None):
    device = get_device(*sequences, device=device)

    sequence = cat_sequence(sequences=sequences, device=device)
    return pack_catted_sequence(sequence=sequence, device=device)


def pack_catted_indices(token_sizes: Tensor, device: torch.device = None):
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(sizes=token_sizes, device=device)
    _, (batch_ptr, token_ptr), batch_sizes = token_sizes_to_minor_ptr3(sizes=token_sizes, batch_ptr=sorted_indices)

    return acc_token_sizes[batch_ptr] + token_ptr, batch_sizes, sorted_indices, unsorted_indices


def pack_catted_sequence(sequence: CattedSequence, device: torch.device = None):
    sequence, token_sizes, device = broadcast_devices(*sequence, device=device)

    indices, batch_sizes, sorted_indices, unsorted_indices = pack_catted_indices(
        token_sizes=token_sizes, device=device,
    )

    return PackedSequence(
        data=sequence[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def pack_padded_indices(token_sizes: Tensor, device: torch.device = None):
    token_sizes, device = broadcast_devices(token_sizes, device=device)

    sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting(sizes=token_sizes, device=device)
    _, (batch_ptr, token_ptr), batch_sizes = token_sizes_to_minor_ptr3(
        sizes=sorted_token_sizes, batch_ptr=sorted_indices,
    )

    return (batch_ptr, token_ptr), batch_sizes, sorted_indices, unsorted_indices


def pack_padded_sequence(sequence: Tensor, token_sizes: Tensor, device: torch.device = None):
    sequence, token_sizes, device = broadcast_devices(sequence, token_sizes, device=device)

    indices, batch_sizes, sorted_indices, unsorted_indices = pack_padded_indices(
        token_sizes=token_sizes, device=device,
    )

    return PackedSequence(
        data=sequence[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
