from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.core import CattedSequence, get_device
from torchrua.info import sequence_info

__all__ = [
    'segment_indices', 'segment_sequence',
]

Sequence = Union[CattedSequence, PackedSequence]


def segment_indices(sizes: Sequence, token_size: int, device: torch.device = None):
    device = get_device(sizes.data, device=device)
    (b, t), (batch_ptr, token_ptr) = sequence_info(sequence=sizes)

    lengths = torch.zeros((b, t + 1), dtype=torch.long, device=device)
    lengths[batch_ptr, token_ptr] = sizes.data
    lengths[:, -1] = token_size - lengths.sum(dim=-1)

    mask = torch.zeros((b, t), dtype=torch.bool, device=device)
    mask[batch_ptr, token_ptr] = True

    return lengths.view(-1), mask, (b, t), (batch_ptr, token_ptr)


def segment_sequence(tensor: Tensor, sizes: Sequence, reduce_fn, keep: bool):
    _, t, *_ = tensor.size()
    tensor = tensor.flatten(start_dim=0, end_dim=1)

    lengths, mask, (b, t), (batch_ptr, token_ptr) = segment_indices(sizes, token_size=t, device=tensor.device)
    data = reduce_fn(tensor, lengths)

    if keep:
        sequence = data.view((b, t + 1, *data.size()[1:]))[:, :-1]
        return sequence, mask, (batch_ptr, token_ptr)

    sequence = sizes._replace(data=data[batch_ptr * (t + 1) + token_ptr])
    return sequence, mask, (batch_ptr, token_ptr)
