from typing import Union, Tuple, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua import get_device, batch_sizes_to_ptr, lengths_to_ptr, accumulate_lengths

__all__ = [
    'pad_sequence',
    'pad_catted_sequence',
    'pad_packed_sequence',
]


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


def pad_sequence(sequences: List[Tensor], batch_first: bool = False,
                 total_length: int = None, padding_value: float = 0) -> Tensor:
    tensor = torch.cat(sequences, dim=0)
    unsorted_lengths = torch.tensor(
        [sequence.size()[0] for sequence in sequences],
        dtype=torch.long, device=tensor.device,
    )

    batch_ptr, token_ptr, _ = lengths_to_ptr(
        lengths=unsorted_lengths,
        sorted_indices=None,
        device=unsorted_lengths.device,
    )

    acc_lengths = accumulate_lengths(lengths=unsorted_lengths)
    indices = acc_lengths[batch_ptr] + token_ptr

    batch_size = len(sequences)
    if total_length is None:
        total_length = unsorted_lengths.max().detach().cpu().item()

    if batch_first:
        data = torch.full(
            (batch_size, total_length, *tensor.size()[1:]),
            fill_value=padding_value, dtype=tensor.dtype, device=tensor.device, requires_grad=False,
        )
        data[batch_ptr, token_ptr] = tensor[indices]
    else:
        data = torch.full(
            (total_length, batch_size, *tensor.size()[1:]),
            fill_value=padding_value, dtype=tensor.dtype, device=tensor.device, requires_grad=False,
        )
        data[token_ptr, batch_ptr] = tensor[indices]

    return data


def pad_catted_sequence():
    raise NotImplementedError
