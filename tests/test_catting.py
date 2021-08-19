import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close, assert_equal, assert_grad_close
from torchrua import catting as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_cat_packed_sequence(data, token_sizes, dim, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = tgt.pack_sequence(inputs, enforce_sorted=False)

    actual_data, actual_token_sizes = rua.cat_sequence(inputs, device=device)
    expected_data, expected_token_sizes = rua.cat_packed_sequence(packed_sequence, device=device)

    assert_close(actual_data, expected_data)
    assert_equal(actual_token_sizes, expected_token_sizes)
    assert_grad_close(actual_data, expected_data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_cat_padded_sequence(data, token_sizes, dim, batch_first, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    padded_sequence = tgt.pad_sequence(inputs, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)

    actual_data, actual_token_sizes = rua.cat_sequence(inputs, device=device)
    expected_data, expected_token_sizes = rua.cat_padded_sequence(
        padded_sequence, token_sizes, batch_first=batch_first, device=device)

    assert_close(actual_data, expected_data)
    assert_equal(actual_token_sizes, expected_token_sizes)
    assert_grad_close(actual_data, expected_data, inputs=inputs)
