import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from torchrua.padding import pack_to_lengths
from torchrua.utils import packed_sequence_to_mask, fetch_batch_sizes, fetch_device, fetch_total_length, \
    fetch_batch_size


@torch.no_grad()
def batch_indices(pack: PackedSequence, unsort: bool = False, total_length: int = None, *,
                  dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)
    batch_size = fetch_batch_size(pack)

    indices = torch.arange(1, batch_size + 1, dtype=dtype, device=device)
    if not unsort and pack.sorted_indices is not None:
        indices = indices[pack.sorted_indices]
    indices = indices[None, :].expand((batch_size, -1)).tril(0)
    indices = indices[fetch_batch_sizes(pack, total_length=total_length) - 1]
    return torch.masked_select(indices, indices != 0) - 1


@torch.no_grad()
def token_indices(pack: PackedSequence, reverse: bool = False, total_length: int = None, *,
                  dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)
    batch_size = fetch_batch_size(pack)
    total_length = fetch_total_length(pack, total_length=total_length)

    indices = torch.arange(1, total_length + 1, dtype=dtype, device=device)
    indices = indices[:, None].expand((-1, batch_size))

    mask = torch.ones((batch_size,), dtype=torch.bool, device=device)
    if pack.sorted_indices is not None:
        mask = mask[pack.sorted_indices]
    mask = mask[None, :].expand((batch_size, -1)).tril(0)
    mask = mask[fetch_batch_sizes(pack, total_length=total_length) - 1]

    if reverse:
        indices = indices.flip(dims=[0]) - (~mask).long().sum(dim=0, keepdim=True)

    return torch.masked_select(indices, mask) - 1


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)
    batch_size = fetch_batch_size(pack)

    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices
    return torch.arange(0, batch_size, dtype=dtype, device=device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def last_indices(pack: PackedSequence, unsort: bool = True, lengths: Tensor = None, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)
    if lengths is None:
        lengths = pack_to_lengths(pack=pack, unsort=False)

    indices = pack.batch_sizes.to(dtype=dtype, device=device).cumsum(dim=0)
    indices = F.pad(indices, [2, 0], value=0)[lengths]
    if unsort and pack.unsorted_indices is not None:
        indices = indices[pack.unsorted_indices] + pack.unsorted_indices
    else:
        indices = indices + torch.arange(indices.size(0), dtype=dtype, device=device)
    return indices


def select_last(pack: PackedSequence, unsort: bool = True, lengths: Tensor = None) -> Tensor:
    return pack.data[last_indices(pack=pack, unsort=unsort, lengths=lengths)]


@torch.no_grad()
def init_indices(pack: PackedSequence, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    mask = packed_sequence_to_mask(pack, unsort=False, dtype=torch.long, device=device or pack.data.device)
    shifted_mask = F.pad(mask[:, 1:], [0, 1])
    init_mask = torch.masked_select(shifted_mask.bool(), mask.bool())
    return torch.arange(0, init_mask.size(0), dtype=dtype, device=device)[init_mask]


def select_init(pack: PackedSequence) -> PackedSequence:
    mask = packed_sequence_to_mask(pack, unsort=False, dtype=torch.long, device=pack.data.device)
    shifted_mask = F.pad(mask[:, 1:], [0, 1])
    init_mask = torch.masked_select(shifted_mask.bool(), mask.bool())
    return PackedSequence(
        data=pack.data[init_mask],
        batch_sizes=shifted_mask.sum(dim=0).to(device=torch.device('cpu')),
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def tail_indices(pack: PackedSequence, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    assert pack.batch_sizes[0] == pack.batch_sizes[1], \
        'some sequences contain only one element, truncating is not allowed.'

    device = fetch_device(pack, device=device)
    return torch.arange(pack.batch_sizes[0].item(), pack.batch_sizes.sum().item(), dtype=dtype, device=device)


def select_tail(pack: PackedSequence) -> PackedSequence:
    assert pack.batch_sizes[0] == pack.batch_sizes[1], \
        'some sequences contain only one element, truncating is not allowed.'

    return PackedSequence(
        data=pack.data[pack.batch_sizes[0].item():],
        batch_sizes=pack.batch_sizes[1:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def reverse_indices(pack: PackedSequence, *,
                    dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)

    batch_ptr = batch_indices(pack=pack, unsort=True, dtype=dtype, device=device)
    token_ptr = token_indices(pack=pack, reverse=True, dtype=dtype, device=device)
    indices = F.pad(pack.batch_sizes.to(dtype=dtype, device=device).cumsum(dim=0), [1, 0], value=0)
    return indices[token_ptr] + batch_ptr


def reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    return PackedSequence(
        data=pack.data[reverse_indices(pack)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )
