import torch
from torch import Tensor

from torchrua.layout import C, L, P, R, Z


def last(self: Z) -> Tensor:
    b, *_ = self.size()

    batch_ptr = torch.arange(b, dtype=torch.long, device=self.data.device)
    token_ptr = self.cat_view().token_sizes - 1

    return self[batch_ptr, token_ptr]


C.last = last
L.last = last
P.last = last
R.last = last
