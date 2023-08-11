import torch
from hypothesis import given
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_catted_sequence_close
from torchnyan import assert_grad_close
from torchnyan import assert_packed_sequence_close
from torchnyan import device
from torchnyan import sizes

from torchrua import cat_sequence
from torchrua import pack_sequence
from torchrua import reverse_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_reverse_catted_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    sequence = cat_sequence(inputs)

    expected = cat_sequence([sequence.flip(dims=[0]) for sequence in inputs])
    actual = reverse_sequence(sequence)

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_reverse_packed_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    sequence = pack_sequence(inputs)

    expected = pack_sequence([sequence.flip(dims=[0]) for sequence in inputs])
    actual = reverse_sequence(sequence)

    assert_packed_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
