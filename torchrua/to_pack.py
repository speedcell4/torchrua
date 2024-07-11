import torch
from torch.nn.utils.rnn import PackedSequence

from torchrua import to_self
from torchrua.core import invert_permutation
from torchrua.layout import C, L, P, T

P.pack = to_self


def pack_t(sequence: T) -> P:
    batch_sizes = torch.ones(sequence.size()[:1], dtype=torch.long).cpu()
    sorted_indices = sequence.data.new_tensor([0], dtype=torch.long)
    unsorted_indices = sequence.data.new_tensor([0], dtype=torch.long)

    return PackedSequence(
        data=sequence.data,
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


T.pack = pack_t


def pack_c(sequence: C) -> P:
    data, token_sizes = sequence
    b, t, *sizes = sequence.size()

    if len(sizes) > 0:
        return sequence[sequence.idx().pack()]

    _, sorting_indices = torch.sort(token_sizes.detach().cpu(), descending=True)
    sorting_indices = sorting_indices.to(device=data.device)
    unsorted_indices = invert_permutation(sorting_indices)

    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = unsorted_indices[batch_ptr]

    tensor = data.new_zeros((t, b))
    tensor[token_ptr, batch_ptr] = data

    mask = torch.zeros_like(tensor, dtype=torch.long)
    mask[token_ptr, batch_ptr] = 1

    return PackedSequence(
        data=tensor[mask.bool()],
        batch_sizes=mask.sum(dim=1).detach().cpu(),
        sorted_indices=sorting_indices,
        unsorted_indices=unsorted_indices,
    )


C.pack = pack_c


def pack_l(sequence: L) -> P:
    return sequence[sequence.idx().pack()]


L.pack = pack_l
