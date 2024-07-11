from numbers import Number
from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from torchrua import C, L, P, R


def cat_tensors(tensors: List[Tensor], token_sizes: List[int]) -> C:
    data = torch.cat(tensors, dim=0)
    token_sizes = data.new_tensor(token_sizes, dtype=torch.long)
    return C(data=data, token_sizes=token_sizes)


C.expected_new = cat_tensors


def pack_tensors(tensors: List[Tensor], token_sizes: List[int]) -> P:
    return pack_sequence(tensors, enforce_sorted=False)


P.expected_new = pack_tensors


def left_aligned_tensors(tensors: List[Tensor], token_sizes: List[int], padding_value: Number = 0) -> L:
    data = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    token_sizes = data.new_tensor(token_sizes, dtype=torch.long)
    return L(data=data, token_sizes=token_sizes)


L.expected_new = left_aligned_tensors


def right_aligned_tensors(tensors: List[Tensor], token_sizes: List[int], padding_value: Number = 0) -> R:
    token_sizes = [tensor.size()[0] for tensor in tensors]
    t = max(token_sizes)

    data = torch.stack([F.pad(x, pad=[0, 0, t - x.size()[0], 0], value=padding_value) for x in tensors], dim=0)
    token_sizes = data.new_tensor(token_sizes, dtype=torch.long)
    return R(data=data, token_sizes=token_sizes)


R.expected_new = right_aligned_tensors


def install():
    pass
