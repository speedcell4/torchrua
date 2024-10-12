import torch
from torch import Tensor

from torchrua.layout import C, L, P, R, Z


def head(self: Z) -> Tensor:
    b, *_ = self.size()

    batch_ptr = torch.arange(b, dtype=torch.long, device=self.data.device)
    token_ptr = torch.zeros_like(batch_ptr)

    return self[batch_ptr, token_ptr]


C.head = head
L.head = head
P.head = head
R.head = head
