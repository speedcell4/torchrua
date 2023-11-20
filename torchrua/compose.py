from typing import List, Union

import torch

from torchrua import C, D, P
from torchrua.core import invert_permutation


def compose(sequences: List[Union[C, D, P]]) -> P:
    offset, data, indices, token_sizes = 0, [], [], []

    for sequence in sequences:
        data.append(sequence._data())

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

    return indices.rua(torch.cat(data, dim=0))
