from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from torchrua.indexing import token_sizes_to_ptr, batch_sizes_to_ptr
from torchrua.utils import accumulate_sizes

__all__ = [
    'cat_sequence',
    'cat_packed_sequence',
    'cat_padded_sequence',
]


def cat_sequence(sequences: List[Tensor], device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    if device is None:
        device = sequences[0].device

    data = torch.cat(sequences, dim=0).to(device=device)
    lengths = torch.tensor([s.size()[0] for s in sequences], dtype=torch.long, device=device)
    return data, lengths


def cat_packed_sequence(pack: PackedSequence, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        if device is None:
            device = pack.data.device

        batch_sizes = pack.batch_sizes.to(device=device)
        acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)
        token_ptr, batch_ptr, sorted_lengths = token_sizes_to_ptr(token_sizes=batch_sizes)

        if pack.unsorted_indices is not None:
            unsorted_lengths = sorted_lengths[pack.unsorted_indices]

        index = acc_batch_sizes[token_ptr] + batch_ptr

    return pack.data[index], unsorted_lengths


def cat_padded_sequence(tensor: Tensor, lengths: Tensor, batch_first: bool,
                        device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        if device is None:
            device = tensor[0].device

        lengths = lengths.to(device=device)
        token_ptr, batch_ptr, _ = batch_sizes_to_ptr(lengths=lengths)

    if batch_first:
        data = tensor[batch_ptr, token_ptr]
    else:
        data = tensor[token_ptr, batch_ptr]

    return data, lengths


if __name__ == '__main__':
    print(cat_packed_sequence(pack_sequence([
        torch.arange(5),
        torch.arange(2),
        torch.arange(3),
    ], enforce_sorted=False)))
