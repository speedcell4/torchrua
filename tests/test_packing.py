import torch
from hypothesis import given
from torch.nn.utils.rnn import pack_sequence as expected_pack_sequence
from torch.nn.utils.rnn import pad_sequence as expected_pad_sequence

from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_grad_close
from torchnyan import assert_sequence_close
from torchnyan import device
from torchnyan import sizes
from torchrua import PaddedSequence
from torchrua import cat_sequence
from torchrua import pack_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_pack_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = pack_sequence(inputs)
    expected = expected_pack_sequence(inputs, enforce_sorted=False)

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_pack_catted_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = cat_sequence(inputs).pack()

    expected = expected_pack_sequence(inputs, enforce_sorted=False)

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_pack_padded_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = PaddedSequence(expected_pad_sequence(inputs, batch_first=True), torch.tensor(token_sizes, device=device))
    actual = actual.pack()

    expected = expected_pack_sequence(inputs, enforce_sorted=False)

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
