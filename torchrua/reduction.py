from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.core import invert_permutation
from torchrua.packing import pack_catted_sequence, pack_catted_indices

__all__ = [
    'reduce_catted_indices', 'reduce_catted_sequences',
]


@torch.no_grad()
def reduce_catted_indices(token_sizes1: Tensor, token_sizes2: Tensor, device: Device = None):
    if device is None:
        device = token_sizes1.device

    indices, batch_sizes, _, unsorted_indices = pack_catted_indices(
        token_sizes=token_sizes1,
        device=device,
    )
    unsorted_indices = pack_catted_sequence(
        sequence=unsorted_indices,
        token_sizes=token_sizes2,
        device=device,
    )
    sorted_indices = invert_permutation(unsorted_indices.data)

    return indices, batch_sizes, sorted_indices, unsorted_indices.data


def reduce_catted_sequences(sequences: List[Tuple[Tensor, Tensor]], device: Device = None) -> PackedSequence:
    if device is None:
        device = sequences[0][0].device

    sequence, token_sizes1, token_sizes2 = zip(*[
        (sequence, token_sizes, token_sizes.size()[0])
        for sequence, token_sizes in sequences
    ])
    sequence = torch.cat(sequence, dim=0).to(device=device)
    token_sizes1 = torch.cat(token_sizes1, dim=0).to(device=device)
    token_sizes2 = torch.tensor(token_sizes2, dtype=torch.long, device=device)

    indices, batch_sizes, sorted_indices, unsorted_indices = reduce_catted_indices(
        token_sizes1=token_sizes1,
        token_sizes2=token_sizes2,
        device=device,
    )

    return PackedSequence(
        data=sequence[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices.data,
    )
