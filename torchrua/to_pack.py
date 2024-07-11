import torch
from torch.nn.utils.rnn import PackedSequence

from torchrua import to_self
from torchrua.core import invert_permutation
from torchrua.layout import C, L, P, R


def cat_to_pack(self: C) -> P:
    b, t, *sizes = self.size()

    if len(sizes) > 0:
        return self[self.idx().pack()]

    data, token_sizes = self
    _, sorting_indices = torch.sort(token_sizes.detach().cpu(), descending=True)
    sorting_indices = sorting_indices.to(device=data.device)
    unsorted_indices = invert_permutation(sorting_indices)

    batch_ptr, token_ptr = self.ptr()
    batch_ptr = unsorted_indices[batch_ptr]

    tensor = data.new_zeros((t, b))
    tensor[token_ptr, batch_ptr] = data

    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask[token_ptr, batch_ptr] = True

    return PackedSequence(
        data=tensor[mask],
        batch_sizes=mask.long().sum(dim=1).detach().cpu(),
        sorted_indices=sorting_indices,
        unsorted_indices=unsorted_indices,
    )


C.pack = cat_to_pack


def left_to_pack(self: L) -> P:
    return self[self.idx().pack()]


L.pack = left_to_pack

P.pack = to_self


def right_to_pack(self: R) -> P:
    return self[self.idx().pack()]


R.pack = right_to_pack
