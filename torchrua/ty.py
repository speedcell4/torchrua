from typing import Any
from typing import List
from typing import NamedTuple
from typing import Type
from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


class CattedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, *args, **kwargs) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.to(*args, **kwargs),
            token_sizes=self.token_sizes.to(*args, **kwargs),
        )

    def cpu(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.cpu(),
            token_sizes=self.token_sizes.cpu(),
        )

    def cuda(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.cuda(),
            token_sizes=self.token_sizes.cuda(),
        )

    def detach(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.detach(),
            token_sizes=self.token_sizes.detach(),
        )


class PaddedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, *args, **kwargs) -> 'PaddedSequence':
        return PaddedSequence(
            data=self.data.to(*args, **kwargs),
            token_sizes=self.token_sizes.to(*args, **kwargs),
        )

    def cpu(self) -> 'PaddedSequence':
        return PaddedSequence(
            data=self.data.cpu(),
            token_sizes=self.token_sizes.cpu(),
        )

    def cuda(self) -> 'PaddedSequence':
        return PaddedSequence(
            data=self.data.cuda(),
            token_sizes=self.token_sizes.cuda(),
        )

    def detach(self) -> 'PaddedSequence':
        return PaddedSequence(
            data=self.data.detach(),
            token_sizes=self.token_sizes.detach(),
        )


T = Tensor
C = CattedSequence
D = PaddedSequence
P = PackedSequence

Sequence = Union[T, C, D, P]

Ts = List[T]
Cs = List[C]
Ds = List[D]
Ps = List[P]

Sequences = Union[Ts, Cs, Ds, Ps]


def is_type(obj: Any, ty: Type) -> bool:
    __origin__ = getattr(ty, '__origin__', None)
    __args__ = getattr(ty, '__args__', [])

    if __origin__ is list:
        if not isinstance(obj, list):
            return False

        return all(is_type(o, __args__[0]) for o in obj)

    if __origin__ is tuple:
        if isinstance(obj, (D, C, P)) or not isinstance(obj, tuple):
            return False

        if len(__args__) == 2 and __args__[1] is ...:
            return all(is_type(o, __args__[0]) for o in obj)

        if len(__args__) == len(obj):
            return all(is_type(o, t) for o, t in zip(obj, __args__))

        return False

    if __origin__ is Union:
        return any(is_type(obj, t) for t in __args__)

    return isinstance(obj, ty)


def cp_idx(sequence: Union[C, P]) -> Union[C, P]:
    n, *_ = sequence.data.size()
    data = torch.arange(n, dtype=torch.long, device=sequence.data.device)
    return sequence._replace(data=data)


def d_idx(sequence: D) -> D:
    (b, t), (batch_ptr, token_ptr), _ = sequence.ptr()
    return sequence._replace(data=token_ptr + batch_ptr * t)


C.idx = cp_idx
D.idx = d_idx
P.idx = cp_idx


def rua(index: Union[C, D, P], sequence: Union[C, D, P]) -> Union[C, D, P]:
    return index._replace(data=sequence.data[index.data])


C.rua = rua
D.rua = rua
P.rua = rua
