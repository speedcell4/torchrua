import torch
from torch import Tensor

RTOL = 1e-5
ATOL = 1e-5


def assert_equal(x: Tensor, y: Tensor) -> None:
    assert torch.equal(x, y), f'{x} != {y}'


def assert_close(x: Tensor, y: Tensor) -> None:
    assert torch.allclose(x, y, rtol=RTOL, atol=ATOL), f'{x} != {y}'
