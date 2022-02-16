import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua import CattedSequence, accumulate_sizes, batch_sizes_to_ptr
from torchrua.core import sizes_to_ptr, transpose_sizes


@torch.no_grad()
def reversed_catted_indices(sequence: CattedSequence) -> Tensor:
    token_sizes = sequence.token_sizes.to(device=sequence.data.device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    token_ptr = token_sizes[batch_ptr] - token_ptr - 1
    return token_ptr + acc_token_sizes[batch_ptr]


def reverse_catted_sequence(sequence: CattedSequence) -> CattedSequence:
    indices = reversed_catted_indices(sequence)
    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes,
    )


@torch.no_grad()
def reverse_packed_indices(batch_sizes: Tensor) -> Tensor:
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    token_ptr, batch_ptr = sizes_to_ptr(sizes=batch_sizes)
    token_sizes = transpose_sizes(sizes=batch_sizes)
    token_ptr = token_sizes[batch_ptr] - token_ptr - 1

    return acc_batch_sizes[token_ptr] + batch_ptr


def reverse_packed_sequence(sequence: PackedSequence) -> PackedSequence:
    device = sequence.data.device

    indices = reverse_packed_indices(batch_sizes=sequence.batch_sizes.to(device=device))
    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
