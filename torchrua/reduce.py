import torch

from torchrua.layout import C, T


def scatter_max(tensor: T, index: T, source: T, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='amax', include_self=include_self, dim=dim)


def scatter_min(tensor: T, index: T, source: T, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='amin', include_self=include_self, dim=dim)


def scatter_sum(tensor: T, index: T, source: T, include_self: bool = False, dim: int = 0):
    return torch.index_add(tensor if include_self else torch.zeros_like(tensor), index=index, source=source, dim=dim)


def scatter_mean(tensor: T, index: T, source: T, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='mean', include_self=include_self, dim=dim)


def scatter_prod(tensor: T, index: T, source: T, include_self: bool = False, dim: int = 0):
    return torch.index_reduce(tensor, index=index, source=source, reduce='prod', include_self=include_self, dim=dim)


def scatter_logsumexp(tensor: T, index: T, source: T, include_self: bool = False, dim: int = 0):
    m = scatter_max(tensor, index=index, source=source, include_self=include_self, dim=dim).detach()

    tensor = (tensor - m).exp()
    source = (source - m[index]).exp()
    return scatter_sum(tensor, index=index, source=source, include_self=include_self, dim=dim).log() + m


def segment_max(tensor: T, segment_sizes: T) -> T:
    m = tensor.min().detach().cpu().item()
    return torch.segment_reduce(tensor, reduce='max', lengths=segment_sizes, unsafe=True, initial=m)


def segment_min(tensor: T, segment_sizes: T) -> T:
    m = tensor.max().detach().cpu().item()
    return torch.segment_reduce(tensor, reduce='min', lengths=segment_sizes, unsafe=True, initial=m)


def segment_sum(tensor: T, segment_sizes: T) -> T:
    return torch.segment_reduce(tensor, reduce='sum', lengths=segment_sizes, unsafe=True, initial=0)


def segment_mean(tensor: T, segment_sizes: T) -> T:
    return torch.segment_reduce(tensor, reduce='mean', lengths=segment_sizes, unsafe=True, initial=0)


def segment_prod(tensor: T, segment_sizes: T) -> T:
    return torch.segment_reduce(tensor, reduce='prod', lengths=segment_sizes, unsafe=True, initial=1)


def segment_logsumexp(tensor: T, segment_sizes: T) -> T:
    m = segment_max(tensor, segment_sizes=segment_sizes).detach()

    tensor = (tensor - torch.repeat_interleave(m, dim=0, repeats=segment_sizes)).exp()
    eps = (segment_sizes == 0).to(dtype=tensor.dtype).view((-1, *[1 for _ in tensor.size()[1:]]))
    return (segment_sum(tensor, segment_sizes=segment_sizes) + eps).log() + m


def segment_head(tensor: T, segment_sizes: T) -> T:
    return C(data=tensor, token_sizes=segment_sizes).head()


def segment_last(tensor: T, segment_sizes: T) -> T:
    return C(data=tensor, token_sizes=segment_sizes).last()
