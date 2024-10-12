from typing import List, Union

import torch
from torch.types import Number

from torchrua.layout import C, L, P, R, T, Z


def cat_pack_split(self: Union[C, P]) -> List[T]:
    data, token_sizes = self.cat()
    split_size_or_sections = token_sizes.detach().cpu().tolist()

    return torch.split(data, split_size_or_sections, dim=0)


C.split = cat_pack_split
P.split = cat_pack_split


def left_split(self: L) -> List[T]:
    _, t, *_ = self.size()

    token_sizes = torch.stack([self.token_sizes, t - self.token_sizes], dim=-1).view(-1)
    split_size_or_sections = token_sizes.detach().cpu().tolist()

    return torch.split(self.data.flatten(start_dim=0, end_dim=1), split_size_or_sections, dim=0)[0::2]


L.split = left_split


def right_split(self: L) -> List[T]:
    _, t, *_ = self.size()

    token_sizes = torch.stack([t - self.token_sizes, self.token_sizes], dim=-1).view(-1)
    split_size_or_sections = token_sizes.detach().cpu().tolist()

    return torch.split(self.data.flatten(start_dim=0, end_dim=1), split_size_or_sections, dim=0)[1::2]


R.split = right_split


def tolist(self: Z) -> List[List[Number]]:
    return [tensor.tolist() for tensor in self.detach().cpu().split()]


C.tolist = tolist
L.tolist = tolist
P.tolist = tolist
R.tolist = tolist
