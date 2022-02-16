import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchrua import CattedSequence, accumulate_sizes, batch_sizes_to_ptr
from torchrua.core import major_sizes_to_ptr, transpose_sizes


@torch.no_grad()
def rolled_catted_indices(sequence: CattedSequence, shifts: int) -> Tensor:
    token_sizes = sequence.token_sizes.to(device=sequence.data.device)
    acc_token_sizes = accumulate_sizes(sizes=token_sizes)
    batch_ptr, token_ptr, _ = batch_sizes_to_ptr(batch_sizes=token_sizes)

    token_ptr = (token_ptr + token_sizes[batch_ptr] - shifts) % token_sizes[batch_ptr]
    return token_ptr + acc_token_sizes[batch_ptr]


def roll_catted_sequence(sequence: CattedSequence, shifts: int) -> CattedSequence:
    indices = rolled_catted_indices(sequence, shifts=shifts)
    return CattedSequence(
        data=sequence.data[indices],
        token_sizes=sequence.token_sizes,
    )


@torch.no_grad()
def roll_packed_indices(batch_sizes: Tensor, shifts: int) -> Tensor:
    acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

    batch_ptr, token_ptr = major_sizes_to_ptr(sizes=batch_sizes)
    token_sizes = transpose_sizes(sizes=batch_sizes)[batch_ptr]
    token_ptr = (token_ptr - shifts + token_sizes) % token_sizes

    return acc_batch_sizes[token_ptr] + batch_ptr


def roll_packed_sequence(sequence: PackedSequence, shifts: int) -> PackedSequence:
    device = sequence.data.device

    indices = roll_packed_indices(batch_sizes=sequence.batch_sizes.to(device=device), shifts=shifts)
    return PackedSequence(
        data=sequence.data[indices],
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
    )
