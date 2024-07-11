from typing import Union

import torch
from torch import Tensor
from torch.distributions.utils import lazy_property

from torchrua import C, L, P, R
from torchrua.core import invert_permutation


def token_sizes(self: P) -> Tensor:
    b, t, *_ = self.size()
    batch_ptr, token_ptr = self.ptr()

    mask = self.data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return mask.sum(dim=1)


P.token_sizes = lazy_property(token_sizes)


def batch_sizes(self: Union[C, L, R]) -> Tensor:
    b, t, *_ = self.size()
    batch_ptr, token_ptr = self.ptr()

    mask = self.data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return mask.sum(dim=0)


C.batch_sizes = lazy_property(batch_sizes)
L.batch_sizes = lazy_property(batch_sizes)
R.batch_sizes = lazy_property(batch_sizes)


def sorted_indices(self: Union[C, L, R]) -> Tensor:
    _, index = torch.sort(self.token_sizes.detach().cpu(), descending=True)
    return index.to(device=self.data.device)


C.sorted_indices = lazy_property(sorted_indices)
L.sorted_indices = lazy_property(sorted_indices)
R.sorted_indices = lazy_property(sorted_indices)


def unsorted_indices(self: Union[C, L, R]) -> Tensor:
    return invert_permutation(self.sorted_indices)


C.unsorted_indices = lazy_property(unsorted_indices)
L.unsorted_indices = lazy_property(unsorted_indices)
R.unsorted_indices = lazy_property(unsorted_indices)
