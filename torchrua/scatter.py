from typing import Tuple

import torch
from torch import Tensor
from torch.types import Device

from torchrua.core import accumulate_sizes
from torchrua.padding import pad_catted_indices


@torch.no_grad()
def scatter_counts(index: Tensor, src: Tensor = None) -> Tensor:
    counts = torch.zeros(index.max().item() + 1, dtype=index.dtype, device=index.device)
    return counts.scatter_add_(dim=0, index=index, src=src or torch.ones_like(index))


@torch.no_grad()
def scatter_index_to_ptr(index: Tensor, device: Device = None) -> Tuple[Tensor, Tensor]:
    index = index.to(device=device)
    sorted_indices = torch.argsort(index, dim=0, descending=False)

    return sorted_indices, accumulate_sizes(sizes=scatter_counts(index=index))


@torch.no_grad()
def scatter_catted_indices(index: Tensor, token_sizes: Tensor, device: Device = None):
    if device is None:
        device = index.device

    (b, t), (batch_ptr, token_ptr) = pad_catted_indices(
        token_sizes=token_sizes,
        batch_first=True, device=device,
    )
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    indices = acc_token_sizes[batch_ptr] + index
    _, indices = torch.unique(indices, dim=0, return_inverse=True)
    indices, offsets = scatter_index_to_ptr(index=indices, device=device)

    token_sizes = torch.zeros((b, t), dtype=torch.long, device=device)
    token_sizes[batch_ptr, index] = 1
    token_sizes = token_sizes.sum(dim=1)

    return indices, offsets, token_sizes


def scatter_add(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=0,
    )
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_mul(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    tensor_view = tensor.view((tensor.size()[0], -1))

    ret, _, _, _ = torch.embedding_bag(
        weight=tensor_view.abs().log(),
        indices=indices, offsets=offsets, mode=0,
    )
    sgn, _, _, _ = torch.embedding_bag(
        weight=tensor_view.sign().neg().add(1.),
        indices=indices, offsets=offsets, mode=0,
    )
    sgn = (sgn % 4).neg().add(1.)

    return (sgn.detach() * ret.exp()).view((ret.size()[0], *tensor.size()[1:]))


def scatter_mean(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=1,
    )
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_max(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=2,
    )
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_min(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    ret, _, _, _ = torch.embedding_bag(
        weight=tensor.neg().view((tensor.size()[0], -1)),
        indices=indices, offsets=offsets, mode=2,
    )
    return ret.neg().view((ret.size()[0], *tensor.size()[1:]))


def scatter_logsumexp(tensor: Tensor, index: Tensor) -> Tensor:
    indices, offsets = scatter_index_to_ptr(index=index, device=tensor.device)
    tensor_view = tensor.view((tensor.size()[0], -1))

    with torch.no_grad():
        m, _, _, _ = torch.embedding_bag(
            weight=tensor_view,
            indices=indices, offsets=offsets, mode=2,
        )

    z, _, _, _ = torch.embedding_bag(
        weight=(tensor_view - m[index]).exp(),
        indices=indices, offsets=offsets, mode=0,
    )
    ret = torch.masked_fill(z, z == 0, 1.).log() + m
    return ret.view((ret.size()[0], *tensor.size()[1:]))


def scatter_log_softmax(tensor: Tensor, index: Tensor) -> Tensor:
    return tensor - scatter_logsumexp(tensor=tensor, index=index)[index]


def scatter_softmax(tensor: Tensor, index: Tensor) -> Tensor:
    return scatter_log_softmax(tensor=tensor, index=index).exp()
