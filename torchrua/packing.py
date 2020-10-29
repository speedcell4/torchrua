import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import invert_permutation

from torchrua import batch_indices, token_indices
from torchrua.utils import lengths_to_batch_sizes


def pack_padded_sequence(input: Tensor, lengths: Tensor,
                         batch_first: bool = False, enforce_sorted: bool = True) -> Tensor:
    batch_sizes = lengths_to_batch_sizes(lengths=lengths, dtype=torch.long, device=torch.device('cpu'))

    if not enforce_sorted:
        sorted_indices = lengths.argsort(dim=0, descending=True)
        unsorted_indices = invert_permutation(sorted_indices)
    else:
        sorted_indices = unsorted_indices = None

    pack = PackedSequence(
        data=None,
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )

    batch_ptr = batch_indices(pack, device=input.data.device)
    token_ptr = token_indices(pack, device=input.data.device)
    if batch_first:
        data = input[batch_ptr, token_ptr]
    else:
        data = input[token_ptr, batch_ptr]

    return PackedSequence(
        data=data,
        batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
