import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from torchrua.utils import pack_to_lengths


@torch.no_grad()
def batch_indices(pack: PackedSequence, unsort: bool = False) -> Tensor:
    indices = torch.arange(1, pack.batch_sizes[0].item() + 1)
    if not unsort and pack.sorted_indices is not None:
        indices = indices[pack.sorted_indices]
    indices = indices[None, :].expand((pack.batch_sizes[0].item(), -1)).tril(0)
    indices = indices[pack.batch_sizes - 1]
    return torch.masked_select(indices, indices != 0) - 1


@torch.no_grad()
def token_indices(pack: PackedSequence, reverse: bool = False) -> Tensor:
    indices = torch.arange(1, pack.batch_sizes.size(0) + 1)
    indices = indices[:, None].expand((-1, pack.batch_sizes[0].item()))

    mask = torch.full((pack.batch_sizes[0].item(),), fill_value=True, dtype=torch.bool)
    if pack.sorted_indices is not None:
        mask = mask[pack.sorted_indices]
    mask = mask[None, :].expand((pack.batch_sizes[0].item(), -1)).tril(0)
    mask = mask[pack.batch_sizes - 1]

    if reverse:
        indices = indices.flip(dims=[0]) - (~mask).long().sum(dim=0, keepdim=True)

    return torch.masked_select(indices, mask) - 1


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    if device is None:
        device = pack.data.device
    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices
    return torch.arange(0, pack.batch_sizes[0].item(), dtype=dtype, device=device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def last_indices(pack: PackedSequence, unsort: bool = True, lengths: Tensor = None, *,
                 dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    if device is None:
        device = pack.data.device
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


def init_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_init(pack: PackedSequence) -> PackedSequence:
    raise NotImplementedError


def tail_indices(pack: PackedSequence) -> Tensor:
    assert pack.batch_sizes[0] == pack.batch_sizes[1], \
        'some sequences contain only one element, truncating is not allowed.'

    return torch.arange(pack.batch_sizes[0].item(), pack.batch_sizes.sum().item(), dtype=torch.long)


def select_tail(pack: PackedSequence) -> PackedSequence:
    assert pack.batch_sizes[0] == pack.batch_sizes[1], \
        'some sequences contain only one element, truncating is not allowed.'

    return PackedSequence(
        data=pack.data[pack.batch_sizes[0]:],
        batch_sizes=pack.batch_sizes[1:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


def reverse_indices(pack: PackedSequence) -> Tensor:
    indices = F.pad(pack.batch_sizes.cumsum(dim=0), [1, 0], value=0)
    batch_ptr = batch_indices(pack=pack, unsort=True)
    token_ptr = token_indices(pack=pack, reverse=True)
    return indices[token_ptr] + batch_ptr


def reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    return PackedSequence(
        data=pack.data[reverse_indices(pack)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )
