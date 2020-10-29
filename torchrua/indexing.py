import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from torchrua.padding import pack_to_lengths
from torchrua.utils import fetch_batch_sizes, fetch_device, fetch_total_length, fetch_batch_size, cum_batch_sizes


@torch.no_grad()
def batch_indices(pack: PackedSequence, unsort: bool = False, total_length: int = None, *,
                  dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)
    batch_size = fetch_batch_size(pack)

    if not unsort and pack.sorted_indices is not None:
        sorted_indices = pack.sorted_indices
    else:
        sorted_indices = ...

    indices = torch.arange(1, batch_size + 1, dtype=dtype, device=device)
    indices = indices[None, sorted_indices].expand((batch_size, -1)).tril(0)
    indices = indices[fetch_batch_sizes(pack, total_length=total_length) - 1]

    mask = indices != 0
    return torch.masked_select(indices, mask) - 1


@torch.no_grad()
def token_indices(pack: PackedSequence, reverse: bool = False, total_length: int = None, *,
                  dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)
    batch_size = fetch_batch_size(pack)
    total_length = fetch_total_length(pack, total_length=total_length)

    indices = torch.arange(1, total_length + 1, dtype=dtype, device=device)
    indices = indices[:, None].expand((-1, batch_size))

    mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device).tril(0)
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
def init_indices(pack: PackedSequence, drop_last_n: int = 1, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)
    total_length = fetch_total_length(pack) - drop_last_n

    batch_ptr = batch_indices(pack=pack, unsort=True, dtype=dtype, device=device,
                              total_length=total_length)
    token_ptr = token_indices(pack=pack, reverse=False, dtype=dtype, device=device, total_length=total_length)
    indices = cum_batch_sizes(pack, device=device)
    return indices[token_ptr] + batch_ptr


def select_init(pack: PackedSequence, drop_last_n: int = 1) -> PackedSequence:
    return PackedSequence(
        data=pack.data[init_indices(pack, drop_last_n=drop_last_n)],
        batch_sizes=pack.batch_sizes[drop_last_n:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def tail_indices(pack: PackedSequence, drop_first_n: int = 1, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    assert pack.batch_sizes[0] == pack.batch_sizes[drop_first_n], \
        'some sequences contain only one element, truncating is not allowed.'

    device = fetch_device(pack, device=device)
    return torch.arange(
        pack.batch_sizes[0].item() * drop_first_n, pack.batch_sizes.sum().item(),
        dtype=dtype, device=device,
    )


def select_tail(pack: PackedSequence, drop_first_n: int = 1) -> PackedSequence:
    assert pack.batch_sizes[0] == pack.batch_sizes[1], \
        'some sequences contain only one element, truncating is not allowed.'

    return PackedSequence(
        data=pack.data[pack.batch_sizes[0].item() * drop_first_n:],
        batch_sizes=pack.batch_sizes[drop_first_n:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def reversed_indices(pack: PackedSequence, *,
                     dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)

    batch_ptr = batch_indices(pack, unsort=True, dtype=dtype, device=device)
    token_ptr = token_indices(pack, reverse=True, dtype=dtype, device=device)
    indices = cum_batch_sizes(pack, device=device)
    return indices[token_ptr] + batch_ptr


def reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    return PackedSequence(
        data=pack.data[reversed_indices(pack)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def rolled_indices(pack: PackedSequence, offset: int, *,
                   dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    device = fetch_device(pack, device=device)

    batch_ptr = batch_indices(pack=pack, unsort=True, dtype=dtype, device=device)
    token_ptr = token_indices(pack=pack, reverse=False, dtype=dtype, device=device)
    lengths = pack_to_lengths(pack, unsort=False, device=device)[batch_ptr]

    indices = cum_batch_sizes(pack, device=device)
    return indices[(token_ptr - offset + lengths) % lengths] + batch_ptr


def roll_packed_sequence(pack: PackedSequence, offset: int) -> PackedSequence:
    return PackedSequence(
        data=pack.data[rolled_indices(pack, offset=offset)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )
