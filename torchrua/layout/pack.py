from typing import Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import get_offsets, major_sizes_to_ptr

P = PackedSequence


def size(self: P) -> Tuple[int, ...]:
    b = self.batch_sizes.max().item()
    t, *_ = self.batch_sizes.size()
    _, *data_size = self.data.size()

    return b, t, *data_size


P.size = size


def ptr_p(self: P) -> Tuple[Tensor, Tensor]:
    data, batch_sizes, _, _ = self
    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes.to(device=data.device))

    return batch_ptr, token_ptr


P.ptr = ptr_p


def idx(self) -> P:
    n, *_ = self.data.size()
    index = torch.arange(n, dtype=torch.long, device=self.data.device)

    return self._replace(data=index)


P.idx = idx


def offsets(self) -> Tensor:
    return get_offsets(self.batch_sizes.to(device=self.data.device))


P.offsets = offsets


def raw(self) -> Tensor:
    return self.data


P.raw = raw
