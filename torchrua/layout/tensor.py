from torch import Tensor

T = Tensor


def raw(self: Tensor) -> Tensor:
    return self


T.raw = raw


def _replace(_: T, data: T) -> T:
    return data


T._replace = _replace
