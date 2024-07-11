from typing import Any, List

import torch

from torchrua import C, L, P, R, T


def tensors_to_cat(tensors: List[T]) -> C:
    data = torch.cat(tensors, dim=0)
    token_sizes = [tensor.size()[0] for tensor in tensors]
    return C(data=data, token_sizes=data.new_tensor(token_sizes, dtype=torch.long))


C.new = tensors_to_cat


def tensors_to_left(tensors: List[T], fill_value: Any = 0) -> L:
    return C.new(tensors).left(fill_value=fill_value)


L.new = tensors_to_left


def tensors_to_pack(tensors: List[T]) -> P:
    return C.new(tensors).pack()


P.new = tensors_to_pack


def tensors_to_right(tensors: List[T], fill_value: Any = 0) -> R:
    return C.new(tensors).left(fill_value=fill_value)


R.new = tensors_to_right
