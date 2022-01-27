from typing import Tuple

import torch
from torch import Tensor

from torchrua.catting import CattedSequence
from torchrua.indexing import batch_sizes_to_ptr


@torch.no_grad()
def repeat_interleave_catted_indices(repeats: Tensor, token_sizes: Tensor) -> Tuple[Tensor, Tensor]:
    index, _, _ = batch_sizes_to_ptr(batch_sizes=repeats)

    batch_ptr, _, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)
    token_sizes = torch.zeros_like(token_sizes).scatter_add_(dim=0, index=batch_ptr, src=repeats)

    return index, token_sizes


def repeat_interleave_catted_sequence(sequence: CattedSequence, repeats: Tensor) -> CattedSequence:
    index, token_sizes = repeat_interleave_catted_indices(repeats, sequence.token_sizes)
    return CattedSequence(data=sequence.data[index], token_sizes=token_sizes)
