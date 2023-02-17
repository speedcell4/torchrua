from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import cat_sequence
from torchrua.core import CattedSequence, invert_permutation
from torchrua.packing import pack_catted_indices

__all__ = [
    'compose_catted_indices', 'compose_catted_sequences',
]


@torch.no_grad()
def compose_catted_indices(token_sizes: List[Tensor], device: Device = None):
    if device is None:
        device = token_sizes[0].device

    token_sizes, sizes = cat_sequence(token_sizes, device=device)

    indices0, batch_sizes, _, unsorted_indices = pack_catted_indices(token_sizes=token_sizes, device=device)
    indices1, _, _, _ = pack_catted_indices(token_sizes=sizes, device=device)
    unsorted_indices = unsorted_indices[indices1]
    sorted_indices = invert_permutation(unsorted_indices, device=device)

    return indices0, batch_sizes, sorted_indices, unsorted_indices


def compose_catted_sequences(sequence: CattedSequence, *sequences: CattedSequence, device: Device = None):
    if device is None:
        device = sequence.data.device

    data, token_sizes = zip(sequence, *sequences)
    data = torch.cat(data, dim=0).to(device=device)

    indices, batch_sizes, sorted_indices, unsorted_indices = compose_catted_indices(
        token_sizes=token_sizes, device=device,
    )

    return PackedSequence(
        data=data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
