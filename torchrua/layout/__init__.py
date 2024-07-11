from typing import Any, Type, Union

from torchrua.layout.cat import *
from torchrua.layout.left import *
from torchrua.layout.pack import *
from torchrua.layout.right import *
from torchrua.layout.tensor import *

CLPR = Union[C, L, P, R]


def empty(self: CLPR) -> CLPR:
    return self._replace(data=self.data.new_tensor(()))


C.empty = empty
L.empty = empty
P.empty = empty
R.empty = empty


def is_type(obj: Any, ty: Type) -> bool:
    __origin__ = getattr(ty, '__origin__', None)
    __args__ = getattr(ty, '__args__', [])

    if __origin__ is list:
        if not isinstance(obj, list):
            return False

        return all(is_type(o, __args__[0]) for o in obj)

    if __origin__ is tuple:
        if isinstance(obj, CLPR.__args__) or not isinstance(obj, tuple):
            return False

        if len(__args__) == 2 and __args__[1] is ...:
            return all(is_type(o, __args__[0]) for o in obj)

        if len(__args__) == len(obj):
            return all(is_type(o, t) for o, t in zip(obj, __args__))

        return False

    if __origin__ is Union:
        return any(is_type(obj, t) for t in __args__)

    return isinstance(obj, ty)
