import functools
from typing import Any
from typing import Union, Tuple, Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence

__all__ = [
    'packed_fn', 'packed_method',
    'Packed', 'PackedMeta',
    'PackedSequential',
]


def packed_fn(fn):
    @functools.wraps(fn)
    def wrap(x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(x):
            return fn(x, *args, **kwargs)
        else:
            data = fn(x.data, *args, **kwargs)
            return PackedSequence(
                data=data,
                batch_sizes=x.batch_sizes,
                sorted_indices=x.sorted_indices,
                unsorted_indices=x.unsorted_indices,
            )

    return wrap


def packed_method(method):
    @functools.wraps(method)
    def wrap(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(x):
            return method(self, x, *args, **kwargs)
        else:
            data = method(self, x.data, *args, **kwargs)
            return PackedSequence(
                data=data,
                batch_sizes=x.batch_sizes,
                sorted_indices=x.sorted_indices,
                unsorted_indices=x.unsorted_indices,
            )

    return wrap


class Packed(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super(Packed, self).__init__()
        self.module = module

    def __repr__(self) -> str:
        return f'Packed{self.module.__repr__()}'

    def forward(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        return packed_fn(self.module)(x, *args, **kwargs)


class PackedMeta(type):
    def __new__(mcs, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]):
        return type(name, bases, {
            **attrs, 'forward': packed_method(attrs.get('forward', bases[0].forward))
        })


class PackedSequential(nn.Sequential, metaclass=PackedMeta):
    pass
