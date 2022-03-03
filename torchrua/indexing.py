import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchrua.catting import CattedSequence
from torchrua.core import major_sizes_to_ptr, transpose_sizes, accumulate_sizes

__all__ = [
    'head_catted_indices', 'head_catted_sequence',
    'last_catted_indices', 'last_catted_sequence',
    'init_catted_indices', 'init_catted_sequence',
    'tail_catted_indices', 'tail_catted_sequence',

    'head_packed_indices', 'head_packed_sequence',
    'last_packed_indices', 'last_packed_sequence',
    'init_packed_indices', 'init_packed_sequence',
    'tail_packed_indices', 'tail_packed_sequence',
]


@torch.no_grad()
def head_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    return accumulate_sizes(sizes=token_sizes.to(device=device))


def head_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = head_catted_indices(
        token_sizes=sequence.token_sizes,
        device=sequence.data.device,
    )

    return sequence.data[indices]


@torch.no_grad()
def last_catted_indices(token_sizes: Tensor, device: Device = None):
    if device is None:
        device = token_sizes.device

    return token_sizes.to(device=device).cumsum(dim=0) - 1


def last_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = last_catted_indices(
        token_sizes=sequence.token_sizes,
        device=sequence.data.device,
    )

    return sequence.data[indices]


@torch.no_grad()
def init_catted_indices(token_sizes: Tensor, n: int = 1, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes = token_sizes - n
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    return token_ptr + acc_token_sizes[batch_ptr], token_sizes


def init_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices, token_sizes = init_catted_indices(
        token_sizes=sequence.token_sizes, n=n,
        device=sequence.data.device,
    )

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=token_sizes,
    )


@torch.no_grad()
def tail_catted_indices(token_sizes: Tensor, n: int = 1, device: Device = None):
    if device is None:
        device = token_sizes.device

    token_sizes = token_sizes.to(device=device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)

    token_sizes = token_sizes - n
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    return token_ptr + acc_token_sizes[batch_ptr] + n, token_sizes


def tail_catted_sequence(sequence: CattedSequence, n: int = 1) -> CattedSequence:
    indices, token_sizes = tail_catted_indices(
        token_sizes=sequence.token_sizes, n=n,
        device=sequence.data.device,
    )

    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=token_sizes,
    )


@torch.no_grad()
def head_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor = None, device: Device = None):
    if device is None:
        device = batch_sizes.device

    if unsorted_indices is not None:
        return unsorted_indices.to(device=device)
    else:
        return torch.arange(batch_sizes[0].item(), device=device)


def head_packed_sequence(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    indices = head_packed_indices(
        batch_sizes=sequence.batch_sizes,
        unsorted_indices=sequence.unsorted_indices if unsort else None,
        device=sequence.data.device,
    )

    return sequence.data[indices]


@torch.no_grad()
def last_packed_indices(batch_sizes: Tensor, unsorted_indices: Tensor = None, device: Device = None):
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr = head_packed_indices(batch_sizes=batch_sizes, unsorted_indices=None, device=device)
    token_ptr = transpose_sizes(sizes=batch_sizes) - 1
    indices = acc_batch_sizes[token_ptr] + batch_ptr

    if unsorted_indices is not None:
        indices = indices[unsorted_indices]
    return indices


def last_packed_sequence(sequence: PackedSequence, unsort: bool = True) -> Tensor:
    indices = last_packed_indices(
        batch_sizes=sequence.batch_sizes,
        unsorted_indices=sequence.unsorted_indices if unsort else None,
        device=sequence.data.device,
    )

    return sequence.data[indices]


@torch.no_grad()
def init_packed_indices(batch_sizes: Tensor, n: int = 1, device: Device = None):
    if device is None:
        device = batch_sizes.device

    batch_sizes = batch_sizes.to(device=device)
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_sizes = batch_sizes[n:]
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)

    return acc_batch_sizes[token_ptr] + batch_ptr, batch_sizes


def init_packed_sequence(sequence: PackedSequence, n: int = 1) -> PackedSequence:
    indices, batch_sizes = init_packed_indices(
        batch_sizes=sequence.batch_sizes, n=n,
        device=sequence.data.device,
    )

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )


@torch.no_grad()
def tail_packed_indices(batch_sizes: Tensor, n: int = 1, device: Device = None):
    if device is None:
        device = batch_sizes.device

    indices = torch.arange(batch_sizes[0].item() * n, batch_sizes.sum().item(), device=device)
    batch_sizes = batch_sizes[n:]
    return indices, batch_sizes


def tail_packed_sequence(sequence: PackedSequence, n: int = 1) -> PackedSequence:
    indices, batch_sizes = tail_packed_indices(
        batch_sizes=sequence.batch_sizes, n=n,
        device=sequence.data.device,
    )

    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
