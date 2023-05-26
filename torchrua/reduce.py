import torch
from torch import Tensor

__all__ = [
    'scatter_max', 'segment_max',
    'scatter_min', 'segment_min',
    'scatter_sum', 'segment_sum',
    'scatter_mean', 'segment_mean',
    'scatter_prod', 'segment_prod',
    'scatter_logsumexp', 'segment_logsumexp',
]


def scatter_max(tensor: Tensor, index: Tensor, source: Tensor, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='amax', include_self=include_self, dim=dim)


def scatter_min(tensor: Tensor, index: Tensor, source: Tensor, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='amin', include_self=include_self, dim=dim)


def scatter_sum(tensor: Tensor, index: Tensor, source: Tensor, include_self: bool = False, dim: int = 0):
    return torch.index_add(tensor if include_self else torch.zeros_like(tensor), index=index, source=source, dim=dim)


def scatter_mean(tensor: Tensor, index: Tensor, source: Tensor, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='mean', include_self=include_self, dim=dim)


def scatter_prod(tensor: Tensor, index: Tensor, source: Tensor, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='prod', include_self=include_self, dim=dim)


def scatter_logsumexp(tensor: Tensor, index: Tensor, source: Tensor, include_self: bool = False, dim: int = 0):
    m = scatter_max(tensor, index=index, source=source, include_self=include_self, dim=dim).detach()

    tensor = (tensor - m).exp()
    source = (source - m[index]).exp()
    return scatter_sum(tensor, index=index, source=source, include_self=include_self, dim=dim).log() + m


def segment_max(tensor: Tensor, segment_sizes: Tensor) -> Tensor:
    m = tensor.min().detach().cpu().item()
    return torch.segment_reduce(tensor, reduce='max', lengths=segment_sizes, unsafe=True, initial=m)


def segment_min(tensor: Tensor, segment_sizes: Tensor) -> Tensor:
    m = tensor.max().detach().cpu().item()
    return torch.segment_reduce(tensor, reduce='min', lengths=segment_sizes, unsafe=True, initial=m)


def segment_sum(tensor: Tensor, segment_sizes: Tensor) -> Tensor:
    return torch.segment_reduce(tensor, reduce='sum', lengths=segment_sizes, unsafe=True, initial=0)


def segment_mean(tensor: Tensor, segment_sizes: Tensor) -> Tensor:
    return torch.segment_reduce(tensor, reduce='mean', lengths=segment_sizes, unsafe=True, initial=0)


def segment_prod(tensor: Tensor, segment_sizes: Tensor) -> Tensor:
    return torch.segment_reduce(tensor, reduce='prod', lengths=segment_sizes, unsafe=True, initial=1)


def segment_logsumexp(tensor: Tensor, segment_sizes: Tensor) -> Tensor:
    m = segment_max(tensor, segment_sizes=segment_sizes).detach()

    tensor = (tensor - torch.repeat_interleave(m, dim=0, repeats=segment_sizes)).exp()
    eps = (segment_sizes == 0).to(dtype=tensor.dtype).view((-1, *[1 for _ in tensor.size()[1:]]))
    return (segment_sum(tensor, segment_sizes=segment_sizes) + eps).log() + m
