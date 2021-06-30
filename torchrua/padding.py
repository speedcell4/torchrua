from typing import Union, Tuple, List, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import cat_sequence
from torchrua.indexing import batch_sizes_to_ptr

__all__ = [
    'pad_sequence',
    'pad_catted_sequence',
    'pad_packed_sequence',
]


def pad_sequence(sequences: List[Tensor], batch_first: bool = False,
                 padding_value: Union[int, float, bool] = 0, device: Optional[torch.device] = None) -> Tensor:
    sequence, token_sizes = cat_sequence(sequences=sequences, device=device)
    return pad_catted_sequence(
        sequence=sequence, token_sizes=token_sizes,
        batch_first=batch_first, padding_value=padding_value,
    )


def pad_packed_sequence(sequence: PackedSequence,
                        batch_first: bool = False,
                        padding_value: Union[int, float, bool] = 0) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        device = sequence.data.device
        batch_sizes = sequence.batch_sizes.to(device=device)

        t = batch_sizes.size()[0]
        b = batch_sizes.max().item()

        token_ptr, batch_ptr, token_sizes = batch_sizes_to_ptr(batch_sizes=batch_sizes)
        if sequence.sorted_indices is not None:
            batch_ptr = sequence.sorted_indices[batch_ptr]
        if sequence.unsorted_indices is not None:
            token_sizes = token_sizes[sequence.unsorted_indices]

    if batch_first:
        tensor = torch.full(
            (b, t, *sequence.data.size()[1:]), fill_value=padding_value,
            dtype=sequence.data.dtype, device=device, requires_grad=False,
        )
        tensor[batch_ptr, token_ptr] = sequence.data
    else:
        tensor = torch.full(
            (t, b, *sequence.data.size()[1:]), fill_value=padding_value,
            dtype=sequence.data.dtype, device=device, requires_grad=False,
        )
        tensor[token_ptr, batch_ptr] = sequence.data

    return tensor, token_sizes.cpu()


def pad_catted_sequence(sequence: Tensor, token_sizes: Tensor,
                        batch_first: bool = False,
                        padding_value: Union[int, float, bool] = 0) -> Tensor:
    with torch.no_grad():
        device = sequence.device
        token_sizes = token_sizes.to(device=device)

        t = token_sizes.max().item()
        b = token_sizes.size()[0]

        batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    if batch_first:
        data = torch.full(
            (b, t, *sequence.size()[1:]), fill_value=padding_value,
            dtype=sequence.dtype, device=sequence.device, requires_grad=False,
        )
        data[batch_ptr, token_ptr] = sequence
    else:
        data = torch.full(
            (t, b, *sequence.size()[1:]), fill_value=padding_value,
            dtype=sequence.dtype, device=sequence.device, requires_grad=False,
        )
        data[token_ptr, batch_ptr] = sequence

    return data
