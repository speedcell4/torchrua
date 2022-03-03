from typing import List, Tuple

import torch
from torch import Tensor
from torch.types import Device

from torchrua.catting import cat_sequence, CattedSequence
from torchrua.core import major_sizes_to_ptr


@torch.no_grad()
def concat_catted_indices(token_sizes: List[Tensor], device: Device = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = token_sizes[0].device

    repeats, batch_sizes = cat_sequence(token_sizes, device=device)
    batch_ptr, _ = major_sizes_to_ptr(sizes=batch_sizes)

    batch_ptr = torch.repeat_interleave(batch_ptr, repeats=repeats)
    _, indices = torch.sort(batch_ptr, stable=True, descending=False)
    _, token_sizes = torch.unique(batch_ptr, sorted=True, return_counts=True)

    return indices, token_sizes


def concat_catted_sequences(sequences: List[CattedSequence]) -> CattedSequence:
    data, token_sizes = zip(*sequences)

    indices, token_sizes = concat_catted_indices(token_sizes, device=data[0].device)
    return CattedSequence(
        data=torch.cat(data, dim=0)[indices],
        token_sizes=token_sizes,
    )
