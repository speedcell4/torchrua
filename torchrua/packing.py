from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import invert_permutation

from torchrua import batch_indices, token_indices
from torchrua.utils import lengths_to_batch_sizes, packed_sequence_to_lengths, fetch_batch_size, fetch_total_length


def pack_padded_sequence(input: Tensor, lengths: Tensor,
                         batch_first: bool = False, enforce_sorted: bool = True) -> Tensor:
    batch_sizes = lengths_to_batch_sizes(lengths=lengths, dtype=torch.long, device=torch.device('cpu'))

    if not enforce_sorted:
        sorted_indices = lengths.argsort(dim=0, descending=True)
        unsorted_indices = invert_permutation(sorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    pack = PackedSequence(
        data=None,
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )

    batch_ptr = batch_indices(pack, device=input.data.device)
    token_ptr = token_indices(pack, device=input.data.device)
    if batch_first:
        data = input[batch_ptr, token_ptr]
    else:
        data = input[token_ptr, batch_ptr]

    return PackedSequence(
        data=data,
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def pad_packed_sequence(pack: PackedSequence, batch_first: bool = False,
                        padding_value: Union[int, float, bool] = 0.0,
                        total_length: int = None) -> Tuple[Tensor, Tensor]:
    batch_size = fetch_batch_size(pack)
    total_length = fetch_total_length(pack, total_length=total_length)

    lengths = packed_sequence_to_lengths(pack=pack, unsort=True, dtype=torch.long, device=torch.device('cpu'))
    batch_ptr = batch_indices(pack, device=pack.data.device)
    token_ptr = token_indices(pack, device=pack.data.device)

    if batch_first:
        data = torch.full(
            (batch_size, total_length), fill_value=padding_value,
            device=pack.data.device, requires_grad=False,
        )
        data[batch_ptr, token_ptr] = pack.data
    else:
        data = torch.full(
            (total_length, batch_size), fill_value=padding_value,
            device=pack.data.device, requires_grad=False,
        )
        data[token_ptr, batch_ptr] = pack.data

    return data, lengths
