from typing import NamedTuple, Tuple

import torch
from torch import Tensor

from torchrua.layout import C, major_sizes_to_ptr


class LeftAlignedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, dtype: torch.dtype = None, device: torch.device = None) -> 'LeftAlignedSequence':
        return LeftAlignedSequence(
            data=self.data.to(dtype=dtype, device=device),
            token_sizes=self.token_sizes.to(device=device),
        )

    def double(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.double)

    def float(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.float)

    def half(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.half)

    def long(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.long)

    def int(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.int)

    def short(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.short)

    def char(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.int8)

    def byte(self) -> 'LeftAlignedSequence':
        return self.to(dtype=torch.uint8)

    def cpu(self) -> 'LeftAlignedSequence':
        return LeftAlignedSequence(
            data=self.data.cpu(),
            token_sizes=self.token_sizes.cpu(),
        )

    def cuda(self) -> 'LeftAlignedSequence':
        return LeftAlignedSequence(
            data=self.data.cuda(),
            token_sizes=self.token_sizes.cuda(),
        )

    def detach(self) -> 'LeftAlignedSequence':
        return LeftAlignedSequence(
            data=self.data.detach(),
            token_sizes=self.token_sizes.detach(),
        )

    def size(self) -> Tuple[int, ...]:
        b, *_ = self.token_sizes.size()
        t = self.token_sizes.max().item()
        _, _, *data_size = self.data.size()

        return b, t, *data_size

    def ptr(self) -> Tuple[Tensor, Tensor]:
        token_ptr, batch_ptr = major_sizes_to_ptr(sizes=self.token_sizes)

        return batch_ptr, token_ptr

    def idx(self) -> C:
        _, t, *_ = self.size()
        batch_ptr, token_ptr = self.ptr()

        return C(data=token_ptr + batch_ptr * t, token_sizes=self.token_sizes)

    def offsets(self) -> Tensor:
        b, t, *_ = self.size()
        return torch.arange(b, dtype=torch.long, device=self.data.device) * t

    def raw(self) -> Tensor:
        return self.data.flatten(start_dim=0, end_dim=1)


L = LeftAlignedSequence
