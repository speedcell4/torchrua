from typing import Tuple

import torch

from torchrua import major_sizes_to_ptr
from torchrua.layout import C, L, P, R


def cat_trunc(self: C, trunc: Tuple[int, int]) -> C:
    data, token_sizes = self

    token_sizes = torch.stack([
        torch.full_like(token_sizes, fill_value=trunc[0]),
        token_sizes - trunc[0] - trunc[1],
        torch.full_like(token_sizes, fill_value=trunc[1]),
    ], dim=-1)

    data = torch.split(data, token_sizes.view(-1).cpu().tolist(), dim=0)

    return C(data=torch.cat(data[1::3]), token_sizes=token_sizes[:, 1])


C.trunc = cat_trunc


def left_trunc(self: L, trunc: Tuple[int, int]) -> L:
    data, token_sizes = self
    _, t, *_ = self.size()

    return L(
        data=data[:, trunc[0]:t - trunc[1]],
        token_sizes=token_sizes - trunc[0] - trunc[1],
    )


L.trunc = left_trunc


def pack_trunc(self: P, trunc: Tuple[int, int]) -> P:
    data, batch_sizes, _, _ = self

    batch_sizes = batch_sizes[trunc[0] + trunc[1]:]
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes.to(device=data.device))
    index = batch_ptr + self.offsets()[token_ptr + trunc[0]]

    return self._replace(data=data[index], batch_sizes=batch_sizes)


P.trunc = pack_trunc


def right_trunc(self: R, trunc: Tuple[int, int]) -> R:
    data, token_sizes = self
    _, t, *_ = self.size()

    return R(
        data=data[:, trunc[0]:t - trunc[1]],
        token_sizes=token_sizes - trunc[0] - trunc[1],
    )


R.trunc = right_trunc
