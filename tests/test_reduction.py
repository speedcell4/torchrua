import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pad_sequence, pack_sequence

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close
from torchrua import tree_reduction_indices, tree_reduce_packed_sequence


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_tree_reduce_packed_sequence(data, token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), requires_grad=True, device=device)
        for token_size in token_sizes
    ]

    packed_sequence = pack_sequence(sequences, enforce_sorted=False)
    reduction_indices = tree_reduction_indices(batch_sizes=packed_sequence.batch_sizes.to(device=device))
    prediction = tree_reduce_packed_sequence(torch.add)(packed_sequence.data, reduction_indices=reduction_indices)

    target = pad_sequence(sequences, batch_first=False).sum(dim=0)[packed_sequence.sorted_indices]

    assert_close(prediction, target)
