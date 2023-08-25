import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.nn.utils.rnn import pack_sequence as expected_pack_sequence
from torch.nn.utils.rnn import pad_sequence as expected_pad_sequence
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_close
from torchnyan import assert_grad_close
from torchnyan import device
from torchnyan import sizes

from torchrua import C
from torchrua import D


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_pad_sequence(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, _ = D.new(inputs)
    expected = expected_pad_sequence(inputs, batch_first=True)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_pad_catted_sequence(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, actual_token_sizes = C.new(inputs).pad()

    expected = expected_pad_sequence(inputs, batch_first=True)
    expected_token_sizes = torch.tensor(token_sizes, device=device)

    assert_close(actual=actual, expected=expected)
    assert_close(actual=actual_token_sizes, expected=expected_token_sizes)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_pad_packed_sequence(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, actual_token_sizes = expected_pack_sequence(inputs, enforce_sorted=False).pad()

    expected = expected_pad_sequence(inputs, batch_first=True)
    expected_token_sizes = torch.tensor(token_sizes, device=device)

    assert_close(actual=actual, expected=expected)
    assert_close(actual=actual_token_sizes, expected=expected_token_sizes)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)
