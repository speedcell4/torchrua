from typing import Any, NamedTuple, Type, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


class CattedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, dtype: torch.dtype = None, device: torch.device = None) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.to(dtype=dtype, device=device),
            token_sizes=self.token_sizes.to(dtype=torch.long, device=device),
        )

    def double(self) -> 'CattedSequence':
        return self.to(dtype=torch.double)

    def float(self) -> 'CattedSequence':
        return self.to(dtype=torch.float)

    def half(self) -> 'CattedSequence':
        return self.to(dtype=torch.half)

    def long(self) -> 'CattedSequence':
        return self.to(dtype=torch.long)

    def int(self) -> 'CattedSequence':
        return self.to(dtype=torch.int)

    def short(self) -> 'CattedSequence':
        return self.to(dtype=torch.short)

    def char(self) -> 'CattedSequence':
        return self.to(dtype=torch.int8)

    def byte(self) -> 'CattedSequence':
        return self.to(dtype=torch.uint8)

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

    def to(self, dtype: torch.dtype = None, device: torch.device = None) -> 'PaddedSequence':
        return PaddedSequence(
            data=self.data.to(dtype=dtype, device=device),
            token_sizes=self.token_sizes.to(dtype=torch.long, device=device),
        )

    def double(self) -> 'PaddedSequence':
        return self.to(dtype=torch.double)

    def float(self) -> 'PaddedSequence':
        return self.to(dtype=torch.float)

    def half(self) -> 'PaddedSequence':
        return self.to(dtype=torch.half)

    def long(self) -> 'PaddedSequence':
        return self.to(dtype=torch.long)

    def int(self) -> 'PaddedSequence':
        return self.to(dtype=torch.int)

    def short(self) -> 'PaddedSequence':
        return self.to(dtype=torch.short)

    def char(self) -> 'PaddedSequence':
        return self.to(dtype=torch.int8)

    def byte(self) -> 'PaddedSequence':
        return self.to(dtype=torch.uint8)

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


def is_type(obj: Any, ty: Type) -> bool:
    __origin__ = getattr(ty, '__origin__', None)
    __args__ = getattr(ty, '__args__', [])

    if __origin__ is list:
        if not isinstance(obj, list):
            return False

        return all(is_type(o, __args__[0]) for o in obj)

    if __origin__ is tuple:
        if isinstance(obj, (C, D, P)) or not isinstance(obj, tuple):
            return False

        if len(__args__) == 2 and __args__[1] is ...:
            return all(is_type(o, __args__[0]) for o in obj)

        if len(__args__) == len(obj):
            return all(is_type(o, t) for o, t in zip(obj, __args__))

        return False

    if __origin__ is Union:
        return any(is_type(obj, t) for t in __args__)

    return isinstance(obj, ty)


def idx_cp(sequence: Union[C, P]) -> Union[C, P]:
    n, *_ = sequence.data.size()
    data = torch.arange(n, dtype=torch.long, device=sequence.data.device)
    return sequence._replace(data=data)


def idx_d(sequence: D) -> C:
    _, token_sizes = sequence
    _, t, *_ = sequence.size()
    batch_ptr, token_ptr = sequence.ptr()
    return CattedSequence(data=token_ptr + batch_ptr * t, token_sizes=token_sizes)


C.idx = idx_cp
D.idx = idx_d
P.idx = idx_cp


def _data_t(sequence: T) -> T:
    return sequence


def _data_d(sequence: D) -> T:
    return sequence.data.flatten(start_dim=0, end_dim=1)


def _data_cp(sequence: Union[C, P]) -> T:
    return sequence.data


T._data = _data_t
C._data = _data_cp
D._data = _data_d
P._data = _data_cp


def _replace_t(_: T, data: T) -> T:
    return data


T._replace = _replace_t


def rua(index: Union[T, C, P], sequence: Union[T, C, D, P], *indices: Union[T, C, P]) -> Union[T, C, P]:
    indices = tuple(index.data for index in (index, *indices))
    return index._replace(data=sequence._data()[indices])


T.rua = rua
C.rua = rua
P.rua = rua


def empty_cdp(sequence: Union[C, D, P]) -> Union[C, D, P]:
    return sequence._replace(data=sequence.data.new_tensor(()))


C.empty = empty_cdp
D.empty = empty_cdp
P.empty = empty_cdp
