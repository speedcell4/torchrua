from numbers import Number
from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def left_aligned_tensors(tensors: List[Tensor], padding_value: Number) -> Tensor:
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def right_aligned_tensors(tensors: List[Tensor], padding_value: Number) -> Tensor:
    token_sizes = [tensor.size()[0] for tensor in tensors]
    t = max(token_sizes)

    return torch.stack([
        F.pad(tensor, pad=[0, 0, t - tensor.size()[0], 0], value=padding_value) for tensor in tensors
    ], dim=0)
