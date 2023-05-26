from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import CattedSequence, get_device, major_sizes_to_ptr, major_sizes_to_shapes

__all__ = [
    'segment_indices', 'segment_sequence',
    'segment_catted_indices',
    'segment_packed_indices',
]

Sequence = Union[CattedSequence, PackedSequence]


def segment_indices(sizes: Sequence, token_size: int, device: torch.device = None):
    if isinstance(sizes, CattedSequence):
        return segment_catted_indices(sizes=sizes, token_size=token_size, device=device)

    if isinstance(sizes, PackedSequence):
        return segment_packed_indices(sizes=sizes, token_size=token_size, device=device)

    raise TypeError(f'type {type(sizes)} is not supported')


def segment_catted_indices(sizes: CattedSequence, token_size: int, device: torch.device = None):
    device = get_device(sizes.data, device=device)

    sizes, token_sizes = sizes
    sizes = sizes.to(device=device)
    token_sizes = token_sizes.to(device=device)

    t, b = major_sizes_to_shapes(sizes=token_sizes)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    lengths = torch.zeros((b, t + 1), dtype=torch.long, device=device)
    lengths[batch_ptr, token_ptr] = sizes
    lengths[:, -1] = token_size - lengths.sum(dim=-1)

    mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    mask[batch_ptr, token_ptr] = True

    return lengths.view(-1), mask, (b, t), (batch_ptr, token_ptr)


def segment_packed_indices(sizes: PackedSequence, token_size: int, device: torch.device = None):
    device = get_device(sizes.data, device=device)

    sizes, batch_sizes, sorted_indices, _ = sizes
    sizes = sizes.to(device=device)
    batch_sizes = batch_sizes.to(device=device)
    sorted_indices = sorted_indices.to(device=device)

    b, t = major_sizes_to_shapes(sizes=batch_sizes)
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    batch_ptr = sorted_indices[batch_ptr]

    lengths = torch.zeros((b, t + 1), dtype=torch.long, device=device)
    lengths[batch_ptr, token_ptr] = sizes
    lengths[:, -1] = token_size - lengths.sum(dim=-1)

    mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    mask[batch_ptr, token_ptr] = True

    return lengths.view(-1), mask, (b, t), (batch_ptr, token_ptr)


def segment_sequence(tensor: Tensor, sizes: Sequence, reduce_fn, keep: bool):
    _, t, *_ = tensor.size()
    tensor = tensor.flatten(start_dim=0, end_dim=1)

    lengths, mask, (b, t), (batch_ptr, token_ptr) = segment_indices(sizes, token_size=t, device=tensor.device)
    data = reduce_fn(tensor, lengths)

    if keep:
        return data.view((b, t + 1, *data.size()[1:]))[:, :-1], mask, (batch_ptr, token_ptr)

    return sizes._replace(data=data[batch_ptr * (t + 1) + token_ptr]), mask, (batch_ptr, token_ptr)
