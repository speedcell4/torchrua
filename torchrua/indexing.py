from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.utils import accumulate_sizes, resize_sizes, batch_sizes_to_token_sizes

__all__ = [
    'batch_sizes_to_ptr',
    'token_sizes_to_ptr',
    'head_indices', 'select_head',
    'last_indices', 'select_last',
    'init_indices', 'select_init',
    'tail_indices', 'select_tail',
    'reversed_indices', 'reverse_packed_sequence',
    'rolled_indices', 'roll_packed_sequence',
]


@torch.no_grad()
def batch_sizes_to_ptr(batch_sizes: Tensor,
                       token_ptr: Optional[Tensor] = None,
                       batch_ptr: Optional[Tensor] = None,
                       batch_first: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    t = batch_sizes.size()[0]
    b = batch_sizes.max().item()

    if token_ptr is None:
        token_ptr = torch.arange(t, device=batch_sizes.device)
    assert token_ptr.size() == (t,)

    if batch_ptr is None:
        batch_ptr = torch.arange(b, device=batch_sizes.device)
    assert batch_ptr.size() == (b,)

    if batch_first:
        bt_mask = batch_ptr[:, None] < batch_sizes[None, :]

        token_ptr = torch.masked_select(token_ptr[None, :], mask=bt_mask)
        batch_ptr = torch.masked_select(batch_ptr[:, None], mask=bt_mask)
        sorted_token_sizes = bt_mask.long().sum(dim=1)
    else:
        tb_mask = batch_ptr[None, :] < batch_sizes[:, None]

        token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
        batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
        sorted_token_sizes = tb_mask.long().sum(dim=0)

    return token_ptr, batch_ptr, sorted_token_sizes


@torch.no_grad()
def token_sizes_to_ptr(token_sizes: Tensor,
                       token_ptr: Optional[Tensor] = None,
                       batch_ptr: Optional[Tensor] = None,
                       batch_first: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    t = token_sizes.max().item()
    b = token_sizes.size()[0]

    if token_ptr is None:
        token_ptr = torch.arange(t, device=token_sizes.device)
    assert token_ptr.size() == (t,)

    if batch_ptr is None:
        batch_ptr = torch.arange(b, device=token_sizes.device)
    assert batch_ptr.size() == (b,)

    if batch_first:
        bt_mask = token_ptr[None, :] < token_sizes[:, None]

        token_ptr = torch.masked_select(token_ptr[None, :], mask=bt_mask)
        batch_ptr = torch.masked_select(batch_ptr[:, None], mask=bt_mask)
        sorted_batch_sizes = bt_mask.long().sum(dim=0)
    else:
        tb_mask = token_ptr[:, None] < token_sizes[None, :]

        token_ptr = torch.masked_select(token_ptr[:, None], mask=tb_mask)
        batch_ptr = torch.masked_select(batch_ptr[None, :], mask=tb_mask)
        sorted_batch_sizes = tb_mask.long().sum(dim=1)

    return token_ptr, batch_ptr, sorted_batch_sizes


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True) -> Tensor:
    device = pack.data.device

    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices.to(device=device)

    b = pack.batch_sizes[0].item()
    return torch.arange(0, b, device=device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def last_indices(pack: PackedSequence, unsort: bool = True) -> Tensor:
    device = pack.data.device

    batch_sizes = pack.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    batch_ptr = head_indices(pack=pack, unsort=unsort)
    token_ptr = batch_sizes_to_token_sizes(batch_sizes=batch_sizes, batch_ptr=batch_ptr) - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_last(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[last_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def init_indices(pack: PackedSequence, drop_last_n: int = 1) -> Tensor:
    device = pack.data.device
    total_length = pack.batch_sizes.size()[0] - drop_last_n

    batch_sizes = pack.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    batch_sizes = resize_sizes(sizes=batch_sizes, n=total_length)
    token_ptr, batch_ptr, _ = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    return acc_batch_sizes[token_ptr] + batch_ptr


def select_init(pack: PackedSequence, drop_last_n: int = 1) -> PackedSequence:
    return PackedSequence(
        data=pack.data[init_indices(pack, drop_last_n=drop_last_n)],
        batch_sizes=pack.batch_sizes[drop_last_n:],
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def tail_indices(pack: PackedSequence, drop_first_n: int = 1) -> Tensor:
    assert pack.batch_sizes[0] == pack.batch_sizes[drop_first_n], \
        f'some sequences contain less than {drop_first_n} elements, truncating is not allowed.'

    device = pack.data.device
    return torch.arange(
        pack.batch_sizes[0].item() * drop_first_n,
        pack.batch_sizes.sum().item(),
        device=device,
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
def reversed_indices(pack: PackedSequence) -> Tensor:
    device = pack.data.device

    batch_sizes = pack.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    token_ptr, batch_ptr, sorted_lengths = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    token_ptr = (sorted_lengths - 1)[batch_ptr] - token_ptr

    return acc_batch_sizes[token_ptr] + batch_ptr


def reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    return PackedSequence(
        data=pack.data[reversed_indices(pack)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )


@torch.no_grad()
def rolled_indices(pack: PackedSequence, offset: int) -> Tensor:
    device = pack.data.device

    batch_sizes = pack.batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
    token_ptr, batch_ptr, sorted_lengths = batch_sizes_to_ptr(batch_sizes=batch_sizes)

    lengths = sorted_lengths[batch_ptr]
    token_ptr = (token_ptr - offset + lengths) % lengths

    return acc_batch_sizes[token_ptr] + batch_ptr


def roll_packed_sequence(pack: PackedSequence, offset: int) -> PackedSequence:
    return PackedSequence(
        data=pack.data[rolled_indices(pack, offset=offset)],
        batch_sizes=pack.batch_sizes,
        sorted_indices=pack.sorted_indices,
        unsorted_indices=pack.unsorted_indices,
    )
