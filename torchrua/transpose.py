from typing import Union

import torch
from torch import Tensor
from torch.distributions.utils import lazy_property

from torchrua import C, L, P, R
from torchrua.core import invert_permutation


def token_sizes(self: P) -> Tensor:
    if not hasattr(self, '_token_sizes'):
        b, t, *_ = self.size()
        batch_ptr, token_ptr = self.ptr()

        mask = self.data.new_zeros((b, t), dtype=torch.long)
        mask[batch_ptr, token_ptr] = 1

        setattr(self, '_token_sizes', mask.sum(dim=1))

    return getattr(self, '_token_sizes')


P.token_sizes = lazy_property(token_sizes)


def batch_sizes(self: Union[C, L, R]) -> Tensor:
    if not hasattr(self, '_batch_sizes'):
        b, t, *_ = self.size()
        batch_ptr, token_ptr = self.ptr()

        mask = self.data.new_zeros((b, t), dtype=torch.long)
        mask[batch_ptr, token_ptr] = 1

        setattr(self, '_batch_sizes', mask.sum(dim=0))

    return getattr(self, '_batch_sizes')


C.batch_sizes = lazy_property(batch_sizes)
L.batch_sizes = lazy_property(batch_sizes)
R.batch_sizes = lazy_property(batch_sizes)


def sorted_indices(self: Union[C, L, R]) -> Tensor:
    if not hasattr(self, '_sorted_indices'):
        _, index = torch.sort(self.token_sizes.detach().cpu(), descending=True)
        setattr(self, '_sorted_indices', index.to(device=self.data.device))

    return getattr(self, '_sorted_indices')


C.sorted_indices = lazy_property(sorted_indices)
L.sorted_indices = lazy_property(sorted_indices)
R.sorted_indices = lazy_property(sorted_indices)


def unsorted_indices(self: Union[C, L, R]) -> Tensor:
    if not hasattr(self, '_unsorted_indices'):
        setattr(self, '_unsorted_indices', invert_permutation(self.sorted_indices))

    return getattr(self, '_unsorted_indices')


C.unsorted_indices = lazy_property(unsorted_indices)
L.unsorted_indices = lazy_property(unsorted_indices)
R.unsorted_indices = lazy_property(unsorted_indices)
