import torch
from torch import Tensor
from torch.distributions.utils import lazy_property

from torchrua import P


def token_sizes(self: P) -> Tensor:
    b, t, *_ = self.size()
    batch_ptr, token_ptr = self.ptr()

    mask = self.data.new_zeros((b, t), dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return mask.sum(dim=1)


P.token_sizes = lazy_property(token_sizes)
