import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua.catting import CattedSequence, cat_packed_indices
from torchrua.packing import pack_catted_indices

__all__ = [
    'repeat_interleave_catted_indices', 'repeat_interleave_catted_sequence',
    'repeat_interleave_packed_indices', 'repeat_interleave_packed_sequence',
]


@torch.no_grad()
def repeat_interleave_catted_indices(repeats: Tensor, token_sizes: Tensor):
    index = torch.repeat_interleave(repeats)

    batch_ptr = torch.repeat_interleave(token_sizes)
    token_sizes = torch.zeros_like(token_sizes).scatter_add_(dim=0, index=batch_ptr, src=repeats)

    return index, token_sizes


def repeat_interleave_catted_sequence(sequence: CattedSequence, repeats: Tensor) -> CattedSequence:
    indices, token_sizes = repeat_interleave_catted_indices(
        repeats=repeats,
        token_sizes=sequence.token_sizes,
    )
    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=token_sizes,
    )


@torch.no_grad()
def repeat_interleave_packed_indices(repeats: Tensor, batch_sizes: Tensor, unsorted_indices: Tensor):
    index1, token_sizes = cat_packed_indices(
        batch_sizes=batch_sizes, device=repeats.device,
        unsorted_indices=unsorted_indices,
    )
    index2, token_sizes = repeat_interleave_catted_indices(repeats=repeats[index1], token_sizes=token_sizes)
    index3, batch_sizes, sorted_indices, unsorted_indices = pack_catted_indices(token_sizes=token_sizes)
    return index1[index2[index3]], batch_sizes, sorted_indices, unsorted_indices


def repeat_interleave_packed_sequence(sequence: PackedSequence, repeats: Tensor) -> PackedSequence:
    indices, batch_sizes, sorted_indices, unsorted_indices = repeat_interleave_packed_indices(
        repeats=repeats,
        batch_sizes=sequence.batch_sizes,
        unsorted_indices=sequence.unsorted_indices,
    )
    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=batch_sizes.detach().cpu(),
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
