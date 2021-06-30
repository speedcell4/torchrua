import torch
from hypothesis import given, strategies as st

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close
from torchrua import pack_sequence, cat_sequence
from torchrua.padding import pad_sequence
from torchrua.tree_reduction import tree_reduce_packed_indices, tree_reduce_padded_indices
from torchrua.tree_reduction import tree_reduce_catted_indices, tree_reduce_sequence


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_tree_reduce_packed_sequence(data, token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    target = pad_sequence(sequences, device=device).sum(dim=0)

    sequence = pack_sequence(sequences, device=device)
    indices = tree_reduce_packed_indices(batch_sizes=sequence.batch_sizes)
    prediction = tree_reduce_sequence(torch.add)(sequence.data, indices)
    prediction = prediction[sequence.unsorted_indices]

    assert_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_tree_reduce_padded_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    target = pad_sequence(sequences, device=device).sum(dim=0)

    sequence = pad_sequence(sequences, device=device, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)
    indices = tree_reduce_padded_indices(token_sizes=token_sizes, batch_first=batch_first)
    prediction = tree_reduce_sequence(torch.add)(sequence.data, indices)

    assert_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_tree_reduce_catted_sequence(data, token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    target = pad_sequence(sequences, device=device).sum(dim=0)

    sequence, token_sizes = cat_sequence(sequences, device=device)
    indices = tree_reduce_catted_indices(token_sizes=token_sizes)
    prediction = tree_reduce_sequence(torch.add)(sequence.data, indices)

    assert_close(prediction, target)
