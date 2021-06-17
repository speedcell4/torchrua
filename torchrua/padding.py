from typing import Union, Tuple, List, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation
from torchrua.indexing import batch_sizes_to_ptr, token_sizes_to_ptr

from torchrua.utils import get_device, accumulate_sizes
from torchrua.catting import cat_sequence

__all__ = [
    'pad_sequence',
    'pad_catted_sequence',
    'pad_packed_sequence',
]


def pad_packed_sequence(pack: PackedSequence, batch_first: bool = False,
                        padding_value: Union[int, float, bool] = 0,
                        total_length: int = None) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
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


def pad_sequence(sequences: List[Tensor], batch_first: bool = False,
                 total_length: int = None, padding_value: float = 0, device: Optional[torch.device] = None) -> Tensor:
    sequence, lengths = cat_sequence(sequences=sequences, device=device)
    return pad_catted_sequence(
        sequence=sequence, lengths=lengths, batch_first=batch_first,
        padding_value=padding_value, total_length=total_length, device=device,
    )


def pad_catted_sequence(sequence: Tensor, lengths: Tensor,
                        batch_first: bool = False, padding_value: float = 0.,
                        total_length: Optional[None] = None, device: Optional[torch.device] = None) -> Tensor:
    with torch.no_grad():
        if device is None:
            device = sequence.device

        unsorted_lengths = lengths.to(device=device)
        batch_size = unsorted_lengths.size()[0]
        if total_length is None:
            total_length = unsorted_lengths.max().detach().cpu().item()

        batch_ptr, token_ptr, _ = token_sizes_to_ptr(
            token_sizes=unsorted_lengths,
            sorted_indices=None,
            device=unsorted_lengths.device,
        )

        acc_lengths = accumulate_sizes(lengths=unsorted_lengths)
        index = invert_permutation(acc_lengths[batch_ptr] + token_ptr)
        batch_ptr = batch_ptr[index]
        token_ptr = token_ptr[index]

    if batch_first:
        data = torch.full(
            (batch_size, total_length, *sequence.size()[1:]), fill_value=padding_value,
            dtype=sequence.dtype, device=sequence.device, requires_grad=False,
        )
        data[batch_ptr, token_ptr] = sequence
    else:
        data = torch.full(
            (total_length, batch_size, *sequence.size()[1:]), fill_value=padding_value,
            dtype=sequence.dtype, device=sequence.device, requires_grad=False,
        )
        data[token_ptr, batch_ptr] = sequence

    return data
