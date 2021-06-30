from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.catting import cat_sequence
from torchrua.indexing import token_sizes_to_ptr
from torchrua.utils import accumulate_sizes, sizes_to_sorting_indices

__all__ = [
    'pack_sequence',
    'pack_catted_sequence',
    'pack_padded_sequence',
    'pack_catted_sequences',
]


def pack_sequence(sequences: List[Tensor], device: Optional[torch.device] = None) -> PackedSequence:
    sequence, token_sizes = cat_sequence(sequences=sequences, device=device)
    return pack_catted_sequence(sequence=sequence, token_sizes=token_sizes)


def pack_padded_sequence(sequence: Tensor, token_sizes: Tensor, batch_first: bool = False) -> PackedSequence:
    with torch.no_grad():
        sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting_indices(
            sizes=token_sizes, descending=True, device=sequence.device,
        )

        token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
            token_sizes=sorted_token_sizes,
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
        sorted_token_sizes, sorted_indices, unsorted_indices = sizes_to_sorting_indices(
            sizes=token_sizes, descending=True, device=sequence.device,
        )

        token_ptr, batch_ptr, batch_sizes = token_sizes_to_ptr(
            token_sizes=sorted_token_sizes,
            batch_ptr=sorted_indices,
        )

        acc_token_sizes = accumulate_sizes(sizes=token_sizes)
        indices = acc_token_sizes[batch_ptr] + token_ptr

    return PackedSequence(
        data=sequence[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def pack_catted_sequences(sequences: List[Tuple[Tensor, Tensor]],
                          device: Optional[torch.device] = None) -> PackedSequence:
    if device is None:
        device = sequences[0][0].device

    sequence, token_sizes, batch_sizes = zip(*[
        (sequence, token_sizes, token_sizes.size()[0])
        for sequence, token_sizes in sequences
    ])
    sequence = torch.cat(sequence, dim=0).to(device=device)
    token_sizes = torch.cat(token_sizes, dim=0).to(device=device)
    batch_sizes = torch.tensor(batch_sizes, dtype=torch.long, device=device)

    sequence = pack_catted_sequence(sequence=sequence, token_sizes=token_sizes)
    unsorted_indices = pack_catted_sequence(sequence=sequence.unsorted_indices, token_sizes=batch_sizes)

    return PackedSequence(
        data=sequence.data,
        batch_sizes=sequence.batch_sizes,
        sorted_indices=invert_permutation(unsorted_indices.data),
        unsorted_indices=unsorted_indices.data,
    )
