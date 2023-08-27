from typing import List
from typing import Union

import torch

from torchrua import C
from torchrua import D
from torchrua import P
from torchrua.core import invert_permutation


def compose(sequences: List[Union[C, D, P]]) -> P:
    offset, data, idx, token_sizes = 0, [], [], []

    for seq in sequences:
        data.append(seq._data())

        i, t = seq.idx().cat()
        idx.append(i + offset)
        token_sizes.append(t)

        offset += data[-1].size()[0]

    token_sizes = C.new(token_sizes)
    unsorted_indices, _, _, _ = token_sizes.idx().pack()

    idx = C(data=torch.cat(idx, dim=0), token_sizes=token_sizes.data).pack()
    unsorted_indices = idx.unsorted_indices[unsorted_indices]
    sorted_indices = invert_permutation(unsorted_indices)

    idx = idx._replace(
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )

    return idx.rua(torch.cat(data, dim=0))
