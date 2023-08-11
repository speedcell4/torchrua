import functools
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import CattedSequence

__all__ = [
    'rua_fn', 'rua_method',
    'RuaModule', 'RuaMeta', 'RuaSequential',
]

Sequence = Union[Tensor, Tuple[Tensor, Any], CattedSequence, PackedSequence]


def rua_fn(fn):
    @functools.wraps(fn)
    def wrap(sequence: Sequence, *args, **kwargs) -> Sequence:
        if torch.is_tensor(sequence):
            return fn(sequence, *args, **kwargs)

        if isinstance(sequence, CattedSequence):
            return CattedSequence(
                data=fn(sequence.data, *args, **kwargs),
                token_sizes=sequence.token_sizes,
            )

        if isinstance(sequence, PackedSequence):
            return PackedSequence(
                data=fn(sequence.data, *args, **kwargs),
                batch_sizes=sequence.batch_sizes,
                sorted_indices=sequence.sorted_indices,
                unsorted_indices=sequence.unsorted_indices,
            )

        return fn(sequence[0], *args, **kwargs), *sequence[1:]

    return wrap


def rua_method(method):
    @functools.wraps(method)
    def wrap(self_or_cls, sequence: Sequence, *args, **kwargs) -> Sequence:
        if torch.is_tensor(sequence):
            return method(self_or_cls, sequence, *args, **kwargs)

        if isinstance(sequence, CattedSequence):
            return CattedSequence(
                data=method(self_or_cls, sequence.data, *args, **kwargs),
                token_sizes=sequence.token_sizes,
            )

        if isinstance(sequence, PackedSequence):
            return PackedSequence(
                data=method(self_or_cls, sequence.data, *args, **kwargs),
                batch_sizes=sequence.batch_sizes,
                sorted_indices=sequence.sorted_indices,
                unsorted_indices=sequence.unsorted_indices,
            )

        return method(self_or_cls, sequence[0], *args, **kwargs), *sequence[1:]

    return wrap


class RuaModule(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super(RuaModule, self).__init__()
        self.module = module

    def __repr__(self) -> str:
        return f'Rua{self.module.__repr__()}'

    def forward(self, x: Sequence, *args, **kwargs) -> Sequence:
        return rua_fn(self.module)(x, *args, **kwargs)


class RuaMeta(type):
    def __new__(mcs, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]):
        return type(name, bases, {
            **attrs, 'forward': rua_method(attrs.get('forward', bases[0].forward))
        })


class RuaSequential(nn.Sequential, metaclass=RuaMeta):
    pass
