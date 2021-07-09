from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, invert_permutation

from torchrua.packing import pack_catted_sequence

__all__ = [
    'reduce_catted_sequences',
]


def reduce_catted_sequences(sequences: List[Tuple[Tensor, Tensor]],
                            device: Optional[torch.device] = None) -> PackedSequence:
    if device is None:
        device = sequences[0][0].device

    sequence, token_sizes, batch_sizes = zip(*[
        (sequence, token_sizes, token_sizes.size()[0])
        for sequence, token_sizes in sequences
    ])
    sequence = torch.cat(sequence, dim=0).to(device=device)
    token_sizes = torch.cat(token_sizes, dim=0).to(device=device)
    batch_sizes = torch.tensor(batch_sizes, dtype=torch.long, device=device)

    sequence = pack_catted_sequence(sequence=sequence, token_sizes=token_sizes)
    unsorted_indices = pack_catted_sequence(sequence=sequence.unsorted_indices, token_sizes=batch_sizes)

    return PackedSequence(
        data=sequence.data,
        batch_sizes=sequence.batch_sizes,
        sorted_indices=invert_permutation(unsorted_indices.data),
        unsorted_indices=unsorted_indices.data,
    )
