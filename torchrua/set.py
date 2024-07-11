from typing import Tuple, Union

from torch import Tensor

from torchrua.layout import C, L, P, R, T, Z

Key = Union[int, Tensor, Tuple[Tensor, Tensor], Z]


def tensor_setitem(self: T, key: Key, value: Tensor) -> None:
    if isinstance(key, Z.__args__):
        self[key.data] = value
        return None

    return super(T, self).__setitem__(key, value)


T.__setitem__ = tensor_setitem


def cat_setitem(self: C, key: Key, value: Tensor) -> None:
    if isinstance(key, Z.__args__):
        self.data[key.data] = value
        return None

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        key = self.offsets()[key[0]] + key[1]

    if isinstance(key, Tensor):
        self.data[key] = value
        return None

    return super(C, self).__setitem__(key, value)


C.__setitem__ = cat_setitem


def left_setitem(self: L, key: Key, value: Tensor) -> None:
    if isinstance(key, Z.__args__):
        self.data.flatten(start_dim=0, end_dim=1)[key.data] = value
        return None

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        self.data[key[0], key[1]] = value
        return None

    if isinstance(key, Tensor):
        self.data.flatten(start_dim=0, end_dim=1)[key] = value
        return None

    return super(L, self).__setitem__(key, value)


L.__setitem__ = left_setitem


def pack_setitem(self: P, key: Key, value: Tensor) -> None:
    if isinstance(key, Z.__args__):
        self.data[key.data] = value
        return None

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        key = self.unsorted_indices[key[0]] + self.offsets()[key[1]]

    if isinstance(key, Tensor):
        self.data[key] = value
        return None

    return super(P, self).__setitem__(key, value)


P.__setitem__ = pack_setitem


def right_setitem(self: R, key: Key, value: Tensor) -> None:
    if isinstance(key, Z.__args__):
        self.data.flatten(start_dim=0, end_dim=1)[key.data] = value
        return None

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        self.data[key[0], self.size()[1] - self.token_sizes[key[0]] + key[1]] = value
        return None

    if isinstance(key, Tensor):
        self.data.flatten(start_dim=0, end_dim=1)[key] = value
        return None

    return super(R, self).__setitem__(key, value)


R.__setitem__ = right_setitem
