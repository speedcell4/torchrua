from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.indexing import batch_sizes_to_ptr, lengths_to_ptr
from torchrua.utils import lengths_to_sorting_indices, get_device

__all__ = [
    'pack_padded_sequence', 'pad_packed_sequence',
]


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


def pad_packed_sequence(pack: PackedSequence, batch_first: bool = False,
                        padding_value: Union[int, float, bool] = 0,
                        total_length: int = None) -> Tuple[Tensor, Tensor]:
    device = get_device(pack)
    batch_size = pack.batch_sizes[0].item()
    if total_length is None:
        total_length = pack.batch_sizes.size(0)

    batch_ptr, token_ptr, lengths = batch_sizes_to_ptr(
        batch_sizes=pack.batch_sizes.to(device=device),
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
        total_length=total_length, device=device,
    )

    if batch_first:
        data = torch.full(
            (batch_size, total_length, *pack.data.size()[1:]),
            fill_value=padding_value, dtype=pack.data.dtype, device=device, requires_grad=False,
        )
        data[batch_ptr, token_ptr] = pack.data
    else:
        data = torch.full(
            (total_length, batch_size, *pack.data.size()[1:]),
            fill_value=padding_value, dtype=pack.data.dtype, device=device, requires_grad=False,
        )
        data[token_ptr, batch_ptr] = pack.data

    return data, lengths.cpu()
