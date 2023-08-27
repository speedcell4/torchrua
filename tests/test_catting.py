import torch
from hypothesis import given
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_grad_close
from torchnyan import assert_sequence_close
from torchnyan import device
from torchnyan import sizes

from torchrua import C
from torchrua import D


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_cat_packed_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = C.new(inputs)
    expected = pack_sequence(inputs, enforce_sorted=False).cat()

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_cat_padded_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = C.new(inputs)
    expected = D(
        data=pad_sequence(inputs, batch_first=True),
        token_sizes=torch.tensor(token_sizes, device=device),
    ).cat()

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
