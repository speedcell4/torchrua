from functools import singledispatch

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_ptr, major_sizes_to_size, accumulate_sizes
from torchrua.wrapper import Sequence

__all__ = [
    'segment_indices',
    'segment_sequence',
    'segment_catted_indices',
    'segment_packed_indices',
]


@singledispatch
def segment_indices(sizes: Sequence, token_size: int, device: Device = None):
    raise TypeError(f'type {type(sizes)} is not supported')


@segment_indices.register
def segment_catted_indices(sizes: CattedSequence, token_size: int, device: Device = None):
    if device is None:
        device = sizes.data.device

    sizes, token_sizes = sizes

    sizes = sizes.to(device=device)
    token_sizes = token_sizes.to(device=device)

    t, b = major_sizes_to_size(sizes=token_sizes)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    out = torch.zeros((b, t + 1), dtype=torch.long, device=device)
    out[batch_ptr, token_ptr] = sizes
    out[:, -1] = token_size - out.sum(dim=-1)

    mask = torch.zeros((b, t + 1), dtype=torch.bool, device=device)
    mask[batch_ptr, token_ptr] = True
    mask[:, -1] = True

    acc_token_sizes = accumulate_sizes(sizes=token_sizes + 1)
    return torch.masked_select(out, mask), acc_token_sizes[batch_ptr] + token_ptr


@segment_indices.register
def segment_packed_indices(sizes: PackedSequence, token_size: int, device: Device = None):
    if device is None:
        device = sizes.data.device

    sizes, batch_sizes, sorted_indices, _ = sizes

    sizes = sizes.to(device=device)
    batch_sizes = batch_sizes.to(device=device)
    sorted_indices = sorted_indices.to(device=device)

    b, t = major_sizes_to_size(sizes=batch_sizes)
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


def segment_sequence(sizes: Sequence, tensor: Tensor, reduce: str, batch_first: bool) -> Sequence:
    if batch_first:
        _, token_size, *_ = tensor.size()
        tensor = rearrange(tensor, 'b t ... -> (b t) ...')
    else:
        token_size, _, *_ = tensor.size()
        tensor = rearrange(tensor, 't b ... -> (b t) ...')

    token_sizes, indices = segment_indices(sizes, token_size=token_size, device=tensor.device)
    data = torch.segment_reduce(tensor, reduce=reduce, lengths=token_sizes, unsafe=True)

    return sizes._replace(data=data[indices])
