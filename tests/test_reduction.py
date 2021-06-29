import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close, assert_grad_close
from torchrua import reduction as rua


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

    packed_sequence = tgt.pack_sequence(sequences, enforce_sorted=False)
    reduction_indices = rua.tree_reduce_packed_indices(batch_sizes=packed_sequence.batch_sizes.to(device=device))
    prediction = rua.tree_reduce_packed_sequence(torch.add)(packed_sequence.data, reduction_indices=reduction_indices)

    target = tgt.pad_sequence(sequences, batch_first=False).sum(dim=0)[packed_sequence.sorted_indices]

    assert_close(prediction, target)
    assert_grad_close(prediction, target, inputs=sequences)
