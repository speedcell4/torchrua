from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import cat_sequence
from torchrua.indexing import token_sizes_to_ptr
from torchrua.utils import sizes_to_sorting_indices, accumulate_sizes

__all__ = [
    'pack_sequence',
    'pack_catted_sequence',
    'pack_padded_sequence',
]


def pack_sequence(sequences: List[Tensor], device: Optional[torch.device] = None) -> PackedSequence:
    sequence, token_sizes = cat_sequence(sequences=sequences, device=device)
    return pack_catted_sequence(sequence=sequence, token_sizes=token_sizes)


def pack_padded_sequence(sequence: Tensor, token_sizes: Tensor, batch_first: bool = False) -> PackedSequence:
    with torch.no_grad():
        device = sequence.device
        token_sizes = token_sizes.to(device=device)

        sorted_indices, unsorted_indices = sizes_to_sorting_indices(sizes=token_sizes)
        token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
            token_sizes=token_sizes,
            batch_ptr=sorted_indices,
        )

        if batch_first:
            index = batch_ptr, token_ptr
        else:
            index = token_ptr, batch_ptr

    return PackedSequence(
        data=sequence[index],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def pack_catted_sequence(sequence: Tensor, token_sizes: Tensor) -> PackedSequence:
    with torch.no_grad():
        device = sequence.device
        token_sizes = token_sizes.to(device=device)

        sorted_indices, unsorted_indices = sizes_to_sorting_indices(sizes=token_sizes)
        token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
            token_sizes=token_sizes,
            batch_ptr=sorted_indices,
        )

        acc_token_sizes = accumulate_sizes(sizes=token_sizes)
        index = acc_token_sizes[batch_ptr] + token_ptr

    return PackedSequence(
        data=sequence[index],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
