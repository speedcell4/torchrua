import torch

from torchrua.layout import C, L, P, T
from torchrua.utils import to_self

C.cat = to_self


def cat_t(sequence: T) -> C:
    token_sizes = sequence.new_tensor(sequence.size()[:1], dtype=torch.long)
    return C(data=sequence, token_sizes=token_sizes)


T.cat = cat_t


def cat_l(sequence: L) -> C:
    return sequence[sequence.idx()]


L.cat = cat_l


def cat_p(sequence: P) -> C:
    data, batch_sizes, sorted_indices, unsorted_indices = sequence
    b, t, *sizes = sequence.size()

    if len(sizes) > 0:
        return sequence[sequence.idx().cat()]

    batch_ptr, token_ptr = sequence.ptr()
    batch_ptr = sorted_indices[batch_ptr]

    tensor = data.new_zeros((b, t))
    tensor[batch_ptr, token_ptr] = data

    mask = torch.zeros_like(tensor, dtype=torch.long)
    mask[batch_ptr, token_ptr] = 1

    return C(
        data=tensor[mask.bool()],
        token_sizes=mask.sum(dim=1),
    )


P.cat = cat_p
