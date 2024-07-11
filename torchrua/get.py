from typing import Tuple, Union

from torch import Tensor

from torchrua.layout import C, L, P, R, Z

Key = Union[int, Tensor, Tuple[Tensor, Tensor], Z]
Value = Union[Tensor, Z]


def cat_getitem(self: C, key: Key) -> Value:
    if isinstance(key, int):
        return super(C, self).__getitem__(key)

    if isinstance(key, Z.__args__):
        return key._replace(data=self.data[key.data])

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        key = self.offsets()[key[0]] + key[1]

    if isinstance(key, Tensor):
        return self.data[key]

    raise NotImplementedError()


C.__getitem__ = cat_getitem


def left_getitem(self: L, key: Key) -> Value:
    if isinstance(key, int):
        return super(L, self).__getitem__(key)

    if isinstance(key, Z.__args__):
        return key._replace(data=self.data.flatten(start_dim=0, end_dim=1)[key.data])

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        return self.data[key[0], key[1]]

    if isinstance(key, Tensor):
        return self.data.flatten(start_dim=0, end_dim=1)[key]

    raise NotImplementedError()


L.__getitem__ = left_getitem


def pack_getitem(self: P, key: Key) -> Value:
    if isinstance(key, int):
        return super(P, self).__getitem__(key)

    if isinstance(key, Z.__args__):
        return key._replace(data=self.data[key.data])

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        key = key[0] + self.offsets()[key[1]]

    if isinstance(key, Tensor):
        return self.data[key]

    raise NotImplementedError()


P.__getitem__ = pack_getitem


def right_getitem(self: R, key: Key) -> Value:
    if isinstance(key, int):
        return super(R, self).__getitem__(key)

    if isinstance(key, Z.__args__):
        return key._replace(data=self.data.flatten(start_dim=0, end_dim=1)[key.data])

    if isinstance(key, tuple) and isinstance(key[0], Tensor) and isinstance(key[1], Tensor):
        return self.data[key[0], self.size()[1] - self.token_sizes[key[0]] + key[1]]

    if isinstance(key, Tensor):
        return self.data.flatten(start_dim=0, end_dim=1)[key]

    raise NotImplementedError()


R.__getitem__ = right_getitem
