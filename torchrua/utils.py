from typing import List

import torch


def with_shape(shape: torch.Size, dim: int, value: int) -> List[int]:
    shape = list(shape)
    shape[dim] = value
    return shape
