import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_packed_sequence_close, assert_grad_close
from torchrua import packing as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_pack_sequence(data, token_sizes, dim, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua.pack_sequence(inputs, device=device)
    excepted = tgt.pack_sequence(inputs, enforce_sorted=False)

    assert_packed_sequence_close(actual, excepted)
    assert_grad_close(actual.data, excepted.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pack_padded_sequence(data, token_sizes, dim, batch_first, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    padded_sequence = tgt.pad_sequence(inputs, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)

    actual = rua.pack_padded_sequence(padded_sequence, token_sizes, batch_first=batch_first)
    excepted = tgt.pack_sequence(inputs, enforce_sorted=False)

    assert_packed_sequence_close(actual, excepted)
    assert_grad_close(actual.data, excepted.data, inputs=inputs)
