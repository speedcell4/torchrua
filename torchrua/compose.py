from typing import List

import torch

from torchrua.layout import C, P, Z
from torchrua.utils import invert_permutation


def compose(sequences: List[Z]) -> P:
    offset, data, indices, token_sizes = 0, [], [], []

    for sequence in sequences:
        data.append(sequence.raw())

        idx, sizes = sequence.idx().cat()
        indices.append(idx + offset)
        token_sizes.append(sizes)

        offset += data[-1].size()[0]

    token_sizes = C.new(token_sizes)
    unsorted_indices, _, _, _ = token_sizes.idx().pack()

    indices = C(data=torch.cat(indices, dim=0), token_sizes=token_sizes.data).pack()
    unsorted_indices = indices.unsorted_indices[unsorted_indices]
    sorted_indices = invert_permutation(unsorted_indices)

    indices = indices._replace(
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )

    return torch.cat(data, dim=0)[indices]
