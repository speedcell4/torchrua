from numbers import Number
from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from torchrua import C, L, P, R


def cat_tensors(tensors: List[Tensor]) -> C:
    data = torch.cat(tensors, dim=0)
    token_sizes = [tensor.size()[0] for tensor in tensors]
    return C(data=data, token_sizes=data.new_tensor(token_sizes, dtype=torch.long))


C.raw_new = cat_tensors


def pack_tensors(tensors: List[Tensor]) -> P:
    return pack_sequence(tensors, enforce_sorted=False)


P.raw_new = pack_tensors


def left_aligned_tensors(tensors: List[Tensor], padding_value: Number) -> Tensor:
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


L.raw_new = left_aligned_tensors


def right_aligned_tensors(tensors: List[Tensor], padding_value: Number) -> Tensor:
    token_sizes = [tensor.size()[0] for tensor in tensors]
    t = max(token_sizes)

    return torch.stack([
        F.pad(tensor, pad=[0, 0, t - tensor.size()[0], 0], value=padding_value) for tensor in tensors
    ], dim=0)


R.raw_new = right_aligned_tensors
