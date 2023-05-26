from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import accumulate_sizes, CattedSequence, get_device, major_sizes_to_ptr, major_sizes_to_shapes

__all__ = [
    'segment_sequence',
    'segment_indices',
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

    out = torch.zeros((b, t + 1), dtype=torch.long, device=device)
    out[batch_ptr, token_ptr] = sizes
    out[:, -1] = token_size - out.sum(dim=-1)

    mask = torch.zeros((b, t + 1), dtype=torch.bool, device=device)
    mask[batch_ptr, token_ptr] = True
    mask[:, -1] = True

    acc_token_sizes = accumulate_sizes(sizes=token_sizes + 1)
    return torch.masked_select(out, mask), acc_token_sizes[batch_ptr] + token_ptr


def segment_packed_indices(sizes: PackedSequence, token_size: int, device: torch.device = None):
    device = get_device(sizes.data, device=device)

    sizes, batch_sizes, sorted_indices, _ = sizes
    sizes = sizes.to(device=device)
    batch_sizes = batch_sizes.to(device=device)
    sorted_indices = sorted_indices.to(device=device)

    b, t = major_sizes_to_shapes(sizes=batch_sizes)
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    batch_ptr = sorted_indices[batch_ptr]

    out = torch.zeros((b, t + 1), dtype=torch.long, device=device)
    out[batch_ptr, token_ptr] = sizes
    out[:, -1] = token_size - out.sum(dim=-1)

    mask = torch.zeros((b, t + 1), dtype=torch.bool, device=device)
    mask[batch_ptr, token_ptr] = True
    mask[:, -1] = True

    acc_token_sizes = accumulate_sizes(sizes=mask.long().sum(dim=-1))
    return torch.masked_select(out, mask), acc_token_sizes[batch_ptr] + token_ptr


def segment_sequence(tensor: Tensor, sizes: Sequence, reduce: str):
    _, token_size, *_ = tensor.size()
    tensor = tensor.flatten(start_dim=0, end_dim=2)

    token_sizes, indices = segment_indices(sizes, token_size=token_size, device=tensor.device)
    data = torch.segment_reduce(tensor, reduce=reduce, lengths=token_sizes, unsafe=True)

    return sizes._replace(data=data[indices])
