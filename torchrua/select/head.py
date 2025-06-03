import torch
from torch import Tensor

from torchrua.layout import C, L, P, R, Z


def cat_head(self: C, n: int = 1) -> C:
    data, token_sizes1 = self
    token_sizes0 = torch.full_like(token_sizes1, fill_value=n)

    split_sizes = torch.stack([token_sizes0, token_sizes1 - token_sizes0], dim=-1).view(-1)
    split_sizes = split_sizes.detach().cpu().tolist()

    data = torch.split_with_sizes(data, split_sizes=split_sizes, dim=0)
    data = torch.cat(data[0::2], dim=0)

    return C(data=data, token_sizes=token_sizes0)


C.head = cat_head


def head(self: Z) -> Tensor:
    b, *_ = self.size()

    batch_ptr = torch.arange(b, dtype=torch.long, device=self.data.device)
    token_ptr = torch.zeros_like(batch_ptr)

    return self[batch_ptr, token_ptr]


L.head = head
P.head = head
R.head = head
