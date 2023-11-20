from typing import Tuple, Union

import torch

from torchrua.core import get_offsets, major_sizes_to_ptr
from torchrua.ty import C, D, P, T, is_type


def size_c(sequence: C) -> Tuple[int, ...]:
    data, token_sizes = sequence
    return token_sizes.size()[0], token_sizes.max().item(), *data.size()[1:]


C.size = size_c


def size_d(sequence: D) -> Tuple[int, ...]:
    data, token_sizes = sequence
    return token_sizes.size()[0], token_sizes.max().item(), *data.size()[2:]


D.size = size_d


def size_p(sequence: P) -> Tuple[int, ...]:
    data, batch_sizes, _, _ = sequence
    return batch_sizes.max().item(), batch_sizes.size()[0], *data.size()[1:]


P.size = size_p


def ptr_cd(sequence: Union[C, D]) -> Tuple[T, T]:
    _, token_sizes = sequence
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)
    return batch_ptr, token_ptr


C.ptr = ptr_cd
D.ptr = ptr_cd


def ptr_p(sequence: P) -> Tuple[T, T]:
    data, batch_sizes, _, _ = sequence
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes.to(device=data.device))
    return batch_ptr, token_ptr


P.ptr = ptr_p


def offsets_c(sequence: C) -> T:
    _, token_sizes = sequence
    return get_offsets(token_sizes)


C.offsets = offsets_c


def offsets_d(sequence: D) -> T:
    data, token_sizes = sequence
    b, t = sequence.size()
    return torch.arange(b, dtype=torch.long, device=data.device) * t


D.offsets = offsets_d


def offsets_p(sequence: P) -> T:
    data, batch_sizes, _, _ = sequence
    return get_offsets(batch_sizes.to(device=data.device))


P.offsets = offsets_p


def __getitem__c(sequence: C, ptr: Tuple[T, T]) -> C:
    if is_type(ptr, Tuple[T, T]):
        batch_ptr, token_ptr = ptr
        return sequence._replace(data=sequence.data[sequence.offsets()[batch_ptr] + token_ptr])

    return super(C, sequence).__getitem__(ptr)


C.__getitem__ = __getitem__c


def __getitem__d(sequence: D, ptr: Tuple[T, T]) -> D:
    if is_type(ptr, Tuple[T, T]):
        batch_ptr, token_ptr = ptr
        return sequence._replace(data=sequence.data[batch_ptr, token_ptr])

    return super(D, sequence).__getitem__(ptr)


D.__getitem__ = __getitem__d


def __getitem__p(sequence: P, ptr: Tuple[T, T]) -> P:
    if is_type(ptr, Tuple[T, T]):
        batch_ptr, token_ptr = ptr
        return sequence._replace(data=sequence.data[batch_ptr + sequence.offsets()[token_ptr]])

    return super(P, sequence).__getitem__(ptr)


P.__getitem__ = __getitem__p
