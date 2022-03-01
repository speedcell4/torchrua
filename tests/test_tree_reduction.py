import torch
from hypothesis import given, strategies as st

from tests.strategies import draw_token_sizes, draw_embedding_dim, draw_device
from tests.utils import assert_close, assert_grad_close
from torchrua import pack_sequence, cat_sequence
from torchrua.padding import pad_sequence
from torchrua.tree_reduction import tree_reduce_catted_indices, tree_reduce_sequence
from torchrua.tree_reduction import tree_reduce_packed_indices, tree_reduce_padded_indices


@given(
    data=st.data(),
    token_sizes=draw_token_sizes(),
    dim=draw_embedding_dim(),
    device=draw_device(),
)
def test_tree_reduce_packed_sequence(data, token_sizes, dim, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted, _ = pad_sequence(inputs, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    packed_sequence = pack_sequence(inputs, device=device)
    indices = tree_reduce_packed_indices(batch_sizes=packed_sequence.batch_sizes)

    actual = tree_reduce_sequence(torch.add)(packed_sequence.data, indices)
    actual = actual[packed_sequence.unsorted_indices]

    assert_close(actual, excepted)
    assert_grad_close(actual, excepted, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=draw_token_sizes(),
    dim=draw_embedding_dim(),
    batch_first=st.booleans(),
    device=draw_device(),
)
def test_tree_reduce_padded_sequence(data, token_sizes, dim, batch_first, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted, _ = pad_sequence(inputs, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    padded_sequence, _ = pad_sequence(inputs, device=device, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)
    indices = tree_reduce_padded_indices(token_sizes=token_sizes, batch_first=batch_first)
    actual = tree_reduce_sequence(torch.add)(padded_sequence, indices)

    assert_close(actual, excepted)
    assert_grad_close(actual, excepted, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=draw_token_sizes(),
    dim=draw_embedding_dim(),
    device=draw_device(),
)
def test_tree_reduce_catted_sequence(data, token_sizes, dim, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    excepted, _ = pad_sequence(inputs, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    catted_sequence = cat_sequence(inputs, device=device)
    indices = tree_reduce_catted_indices(token_sizes=catted_sequence.token_sizes)
    actual = tree_reduce_sequence(torch.add)(catted_sequence.data, indices)

    assert_close(actual, excepted)
    assert_grad_close(actual, excepted, inputs=inputs)
