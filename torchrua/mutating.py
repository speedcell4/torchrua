from typing import Optional, List, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def cat_packed_sequences(packs: List[PackedSequence]) -> PackedSequence:
    data = cat_packed_data(packs=packs)
    batch_sizes, sorted_indices, unsorted_indices = cat_packed_batch_sizes(pack=packs[0], num_packs=len(packs))

    return PackedSequence(
        data=data, batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def cat_packed_data(packs: List[PackedSequence]) -> Tensor:
    data = torch.stack([pack.data for pack in packs], dim=1)
    return rearrange(data, 'p n ... -> (p n) ...')


@torch.no_grad()
def cat_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> \
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    num_batches = pack.batch_sizes[0].item()
    if pack.sorted_indices is not None:
        sorted_indices = torch.stack([
            pack.sorted_indices + num_batches * index
            for index in range(num_packs)
        ], dim=1).view(-1)
    else:
        sorted_indices = None

    if pack.unsorted_indices is not None:
        unsorted_indices = torch.cat([
            pack.unsorted_indices * num_packs + index
            for index in range(num_packs)
        ], dim=0)
    else:
        unsorted_indices = None

    return pack.batch_sizes * num_packs, sorted_indices, unsorted_indices


def uncat_packed_sequence(pack: PackedSequence, num_packs: int) -> List[PackedSequence]:
    batch_sizes, sorted_indices, unsorted_indices = uncat_packed_batch_sizes(pack=pack, num_packs=num_packs)
    return [
        PackedSequence(
            data=data, batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )
        for data in uncat_packed_data(pack=pack, num_packs=num_packs)
    ]


def uncat_packed_data(pack: PackedSequence, num_packs: int) -> List[Tensor]:
    data = rearrange(pack.data, '(p n) ... -> p n ...', n=num_packs)
    return [data[:, index] for index in range(num_packs)]


@torch.no_grad()
def uncat_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> \
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.sorted_indices is None:
        sorted_indices = pack.sorted_indices[::num_packs]
    else:
        sorted_indices = None

    num_batches = pack.batch_sizes[0].item()
    if pack.unsorted_indices is None:
        unsorted_indices = pack.unsorted_indices[:num_batches] // num_packs
    else:
        unsorted_indices = None

    return pack.batch_sizes // num_packs, sorted_indices, unsorted_indices


def stack_packed_sequences(packs: List[PackedSequence]) -> PackedSequence:
    data = stack_packed_data(packs=packs)
    batch_sizes, sorted_indices, unsorted_indices = stack_packed_batch_sizes(pack=packs[0], num_packs=len(packs))

    return PackedSequence(
        data=data, batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


def stack_packed_data(packs: List[PackedSequence]) -> Tensor:
    data = torch.stack([pack.data for pack in packs], dim=1)
    return rearrange(data, 'p n ... -> (p n) ...')


@torch.no_grad()
def stack_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> \
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    num_batches = pack.batch_sizes[0].item()

    if pack.sorted_indices is not None:
        sorted_indices = torch.stack([
            pack.sorted_indices + num_batches * index
            for index in range(num_packs)
        ], dim=1).view(-1)
    else:
        sorted_indices = None

    if pack.unsorted_indices is not None:
        unsorted_indices = torch.stack([
            pack.unsorted_indices * num_packs + index
            for index in range(num_packs)
        ], dim=1).view(-1)
    else:
        unsorted_indices = None

    return pack.batch_sizes * num_packs, sorted_indices, unsorted_indices


def unstack_packed_sequence(pack: PackedSequence, num_packs: int) -> List[PackedSequence]:
    batch_sizes, sorted_indices, unsorted_indices = unstack_packed_batch_sizes(pack=pack, num_packs=num_packs)
    return [
        PackedSequence(
            data=data, batch_sizes=batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )
        for data in unstack_packed_data(pack=pack, num_packs=num_packs)
    ]


def unstack_packed_data(pack: PackedSequence, num_packs: int) -> List[Tensor]:
    data = rearrange(pack.data, '(p n) ... -> p n ...', n=num_packs)
    return [data[:, index] for index in range(num_packs)]


@torch.no_grad()
def unstack_packed_batch_sizes(pack: PackedSequence, num_packs: int) -> \
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if pack.sorted_indices is None:
        sorted_indices = pack.sorted_indices[::num_packs]
    else:
        sorted_indices = None

    if pack.unsorted_indices is None:
        unsorted_indices = pack.unsorted_indices[::num_packs] // num_packs
    else:
        unsorted_indices = None

    return pack.batch_sizes // num_packs, sorted_indices, unsorted_indices
