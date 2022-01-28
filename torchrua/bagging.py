import torch
from torch import Tensor
from torch.types import Device

from torchrua import scatter_index_to_ptr
from torchrua.catting import CattedSequence
from torchrua.padding import pad_catted_indices
from torchrua.utils import accumulate_sizes


@torch.no_grad()
def scatter_catted_indices(index: Tensor, token_sizes: Tensor, device: Device = None):
    if device is None:
        device = index.device

    (b, t), (batch_ptr, token_ptr) = pad_catted_indices(
        token_sizes=token_sizes,
        batch_first=True, device=device,
    )
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    indices = acc_token_sizes[batch_ptr] + index
    _, indices = torch.unique(indices, dim=0, return_inverse=True)
    indices, offsets = scatter_index_to_ptr(index=indices, device=device)

    token_sizes = torch.zeros((b, t), dtype=torch.long, device=device)
    token_sizes[batch_ptr, index] = 1
    token_sizes = token_sizes.sum(dim=1)

    return indices, offsets, token_sizes


def scatter_catted_sequence(sequence: CattedSequence, index: Tensor, mode: int = 0) -> CattedSequence:
    indices, offsets, token_sizes = scatter_catted_indices(
        index=index,
        token_sizes=sequence.token_sizes,
        device=sequence.data.device,
    )
    data, _, _, _ = torch.embedding_bag(
        weight=sequence.data.view((sequence.data.size()[0], -1)),
        indices=indices, offsets=offsets, mode=mode,
    )
    return CattedSequence(data=data.view((-1, *sequence.data.size()[1:])), token_sizes=token_sizes)
