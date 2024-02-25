from typing import Tuple

import torch

from torchrua.core import major_sizes_to_ptr
from torchrua.ty import C, D, P


def trunc_c(sequence: C, trunc: Tuple[int, int]) -> C:
    data, token_sizes = sequence

    token_sizes = torch.stack([
        torch.full_like(token_sizes, fill_value=trunc[0]),
        token_sizes - trunc[0] - trunc[1],
        torch.full_like(token_sizes, fill_value=trunc[1]),
    ], dim=-1)

    data = torch.split(data, token_sizes.view(-1).cpu().tolist(), dim=0)

    return C(data=torch.cat(data[1::3]), token_sizes=token_sizes[:, 1])


C.trunc = trunc_c


def trunc_d(sequence: D, trunc: Tuple[int, int]) -> D:
    data, token_sizes = sequence
    _, t, *_ = sequence.size()

    return D(
        data=data[:, trunc[0]:t - trunc[1]],
        token_sizes=token_sizes - trunc[0] - trunc[1],
    )


D.trunc = trunc_d


def trunc_p(sequence: P, trunc: Tuple[int, int]) -> P:
    data, batch_sizes, _, _ = sequence

    batch_sizes = batch_sizes[trunc[0] + trunc[1]:]
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes.to(device=data.device))
    index = batch_ptr + sequence.offsets()[token_ptr + trunc[0]]

    return sequence._replace(data=data[index], batch_sizes=batch_sizes)


P.trunc = trunc_p
