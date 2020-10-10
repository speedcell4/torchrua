from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import invert_permutation
from torch.nn.utils.rnn import pad_packed_sequence


@torch.no_grad()
def pack_to_mask(pack: PackedSequence, unsort: bool = True, *,
                 padding_value: Union[int, float, bool] = False, batch_first: bool = False,
                 dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if device is None:
        device = pack.data.device

    sorted_indices = None
    if unsort and pack.sorted_indices is not None:
        sorted_indices = pack.sorted_indices

    unsorted_indices = None
    if unsort and pack.unsorted_indices is not None:
        unsorted_indices = pack.unsorted_indices

    mask, _ = pad_packed_sequence(
        PackedSequence(
            data=torch.ones(pack.data.size()[:1], dtype=dtype, device=device),
            batch_sizes=pack.batch_sizes,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        ),
        batch_first=batch_first, padding_value=padding_value,
    )
    return mask


@torch.no_grad()
def pack_to_lengths(pack: PackedSequence, unsort: bool = True, *,
                    dtype: torch.dtype = torch.long, device: torch.device = None) -> Tensor:
    if device is None:
        device = pack.data.device

    batch_size = pack.batch_sizes[0].item()
    mask = torch.ones((batch_size, batch_size), dtype=dtype, device=device).tril(0)
    lengths = mask[pack.batch_sizes - 1].sum(dim=0)

    if unsort and pack.unsorted_indices is not None:
        lengths = lengths[pack.unsorted_indices]
    return lengths


@torch.no_grad()
def lengths_to_mask(lengths: Tensor, filling_mask: bool = True, *,
                    batch_first: bool = False,
                    dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    max_sent_length = lengths.max().item()
    if device is None:
        device = lengths.device

    indices = torch.arange(max_sent_length, dtype=lengths.dtype, device=lengths.device)

    if filling_mask:
        op = torch.lt
    else:
        op = torch.ge

    if batch_first:
        mask = op(indices[None, :], lengths[:, None])
    else:
        mask = op(indices[:, None], lengths[None, :])

    return mask.to(dtype=dtype, device=device)


@torch.no_grad()
def lengths_to_batch_sizes(lengths: Tensor, *, device: torch.device = None) -> Tuple[Tensor, Tensor, Tensor]:
    max_sent_length = lengths.max().item()
    if device is None:
        device = lengths.device

    mask = torch.ones((max_sent_length,), dtype=torch.long, device=device)
    mask = mask[:, None].expand((-1, max_sent_length)).tril(0)

    batch_sizes = mask[lengths - 1].sum(dim=0)
    sorted_indices = lengths.argsort(dim=0, descending=True)
    unsorted_indices = invert_permutation(sorted_indices)

    return batch_sizes, sorted_indices, unsorted_indices
