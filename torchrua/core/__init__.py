from typing import Any, List

from torchrua.core.cast import *
from torchrua.core.get import *
from torchrua.core.set import *
from torchrua.core.view import *


def new_cat(tensors: List[T]) -> C:
    data = torch.cat(tensors, dim=0)
    token_sizes = [tensor.size()[0] for tensor in tensors]
    return C(data=data, token_sizes=data.new_tensor(token_sizes, dtype=torch.long))


C.new = new_cat


def new_left(tensors: List[T], fill_value: Any = 0) -> L:
    return C.new(tensors).left(fill_value=fill_value)


L.new = new_left


def new_pack(tensors: List[T]) -> P:
    return C.new(tensors).pack()


P.new = new_pack


def new_right(tensors: List[T], fill_value: Any = 0) -> R:
    return C.new(tensors).right(fill_value=fill_value)


R.new = new_right
