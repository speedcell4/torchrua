from typing import NamedTuple, Tuple

import torch
from torch import Tensor

from torchrua.utils import get_offsets, major_sizes_to_ptr


class CattedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, dtype: torch.dtype = None, device: torch.device = None) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.to(dtype=dtype, device=device),
            token_sizes=self.token_sizes.to(device=device),
        )

    def double(self) -> 'CattedSequence':
        return self.to(dtype=torch.double)

    def float(self) -> 'CattedSequence':
        return self.to(dtype=torch.float)

    def half(self) -> 'CattedSequence':
        return self.to(dtype=torch.half)

    def long(self) -> 'CattedSequence':
        return self.to(dtype=torch.long)

    def int(self) -> 'CattedSequence':
        return self.to(dtype=torch.int)

    def short(self) -> 'CattedSequence':
        return self.to(dtype=torch.short)

    def char(self) -> 'CattedSequence':
        return self.to(dtype=torch.int8)

    def byte(self) -> 'CattedSequence':
        return self.to(dtype=torch.uint8)

    def cpu(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.cpu(),
            token_sizes=self.token_sizes.cpu(),
        )

    def cuda(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.cuda(),
            token_sizes=self.token_sizes.cuda(),
        )

    def detach(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.detach(),
            token_sizes=self.token_sizes.detach(),
        )

    def size(self) -> Tuple[int, ...]:
        b, *_ = self.token_sizes.size()
        t = self.token_sizes.max().item()
        _, *data_size = self.data.size()

        return b, t, *data_size

    def ptr(self) -> Tuple[Tensor, Tensor]:
        token_ptr, batch_ptr = major_sizes_to_ptr(sizes=self.token_sizes)

        return batch_ptr, token_ptr

    def idx(self) -> 'CattedSequence':
        n, *_ = self.data.size()
        index = torch.arange(n, dtype=torch.long, device=self.data.device)

        return self._replace(data=index)

    def offsets(self) -> Tensor:
        n, *_ = self.data.size()
        return get_offsets(sizes=self.token_sizes).clamp_max_(n - 1)

    def raw(self) -> Tensor:
        return self.data


C = CattedSequence
