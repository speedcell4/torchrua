import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


@torch.no_grad()
def batch_indices(pack: PackedSequence) -> Tensor:
    indices = torch.arange(1, pack.batch_sizes[0].item() + 1)
    indices = indices[None, :].expand((pack.batch_sizes[0].item(), -1)).tril(0)
    indices = indices[pack.batch_sizes - 1]
    return torch.masked_select(indices, indices != 0) - 1


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True) -> Tensor:
    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices
    return torch.arange(0, pack.batch_sizes[0].item(), dtype=torch.long, device=pack.data.device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


def last_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_last(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def init_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_init(pack: PackedSequence) -> PackedSequence:
    raise NotImplementedError


def tail_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_tail(pack: PackedSequence) -> PackedSequence:
    raise NotImplementedError


def reverse_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def flip_packed_sequence(pack: PackedSequence) -> PackedSequence:
    raise NotImplementedError
