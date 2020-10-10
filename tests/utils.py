import torch
from torch import Tensor

from tests.strategies import ATOL, RTOL


def assert_equal(lhs: Tensor, rhs: Tensor) -> None:
    if lhs.dtype == torch.float32 and rhs.dtype == torch.float32:
        assert torch.allclose(lhs, rhs, rtol=RTOL, atol=ATOL), f'{lhs.contiguous().view(-1)} != {rhs.contiguous().view(-1)}'
    elif lhs.dtype == torch.long and rhs.dtype == torch.long:
        assert torch.equal(lhs, rhs), f'{lhs.contiguous().view(-1)} != {rhs.contiguous().view(-1)}'
    elif lhs.dtype == torch.bool and rhs.dtype == torch.bool:
        assert torch.equal(lhs, rhs), f'{lhs.contiguous().view(-1)} != {rhs.contiguous().view(-1)}'
    else:
        raise NotImplementedError
