import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close, assert_equal, assert_grad_close
from torchrua import padding as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pad_sequence(data, token_sizes, dim, batch_first, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, _ = rua.pad_sequence(inputs, batch_first=batch_first)
    excepted = tgt.pad_sequence(inputs, batch_first=batch_first)

    assert_close(actual, excepted)
    assert_grad_close(actual, excepted, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pad_packed_sequence(data, token_sizes, dim, batch_first, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = tgt.pack_sequence(inputs, enforce_sorted=False)
    excepted_token_sizes = torch.tensor(token_sizes, device=device)

    excepted = tgt.pad_sequence(inputs, batch_first=batch_first)
    actual, actual_token_sizes = rua.pad_packed_sequence(packed_sequence, batch_first=batch_first)

    assert_close(actual, excepted)
    assert_grad_close(actual, excepted, inputs=inputs)
    assert_equal(actual_token_sizes, excepted_token_sizes)
