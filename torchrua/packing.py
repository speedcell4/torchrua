from typing import List

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import cat_sequence, pack_catted_sequence
from torchrua.indexing import lengths_to_ptr
from torchrua.utils import lengths_to_sorting_indices, get_device


def pack_padded_sequence(input: Tensor, lengths: Tensor,
                         batch_first: bool = False, enforce_sorted: bool = True) -> PackedSequence:
    device = get_device(input)

    if not enforce_sorted:
        sorted_indices, unsorted_indices = lengths_to_sorting_indices(lengths)
    else:
        sorted_indices = unsorted_indices = None

    batch_ptr, token_ptr, batch_sizes = lengths_to_ptr(
        lengths, sorted_indices=sorted_indices, device=device,
    )

    if batch_first:
        data = input[batch_ptr, token_ptr]
    else:
        data = input[token_ptr, batch_ptr]

    return PackedSequence(
        data=data,
        batch_sizes=batch_sizes.cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def pack_sequence(sequences: List[Tensor]) -> PackedSequence:
    return pack_catted_sequence(sequence=cat_sequence(sequences=sequences))
