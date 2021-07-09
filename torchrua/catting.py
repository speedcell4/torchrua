from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.indexing import token_sizes_to_ptr, batch_sizes_to_ptr
from torchrua.utils import accumulate_sizes

__all__ = [
    'CattedSequence', 'cat_sequence',
    'cat_packed_sequence',
    'cat_padded_sequence',
]

CattedSequence = Tuple[Tensor, Tensor]


def cat_sequence(sequences: List[Tensor], device: Device = None) -> CattedSequence:
    if device is None:
        device = sequences[0].device

    sequence = torch.cat(sequences, dim=0).to(device=device)
    token_sizes = torch.tensor(
        [seq.size()[0] for seq in sequences],
        dtype=torch.long, device=device,
    )
    return sequence, token_sizes


def cat_packed_sequence(sequence: PackedSequence, device: Device = None) -> CattedSequence:
    with torch.no_grad():
        if device is None:
            device = sequence.data.device

        batch_sizes = sequence.batch_sizes.to(device=device)
        batch_ptr, token_ptr, token_sizes = token_sizes_to_ptr(
            token_sizes=batch_sizes,
            token_ptr=sequence.unsorted_indices,
        )
        acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

        indices = acc_batch_sizes[token_ptr] + batch_ptr

    return sequence.data[indices], token_sizes


def cat_padded_sequence(sequence: Tensor, token_sizes: Tensor, batch_first: bool = False,
                        device: Device = None) -> CattedSequence:
    with torch.no_grad():
        if device is None:
            device = sequence[0].device

        token_sizes = token_sizes.to(device=device)
        batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    if batch_first:
        return sequence[batch_ptr, token_ptr], token_sizes
    else:
        return sequence[token_ptr, batch_ptr], token_sizes
