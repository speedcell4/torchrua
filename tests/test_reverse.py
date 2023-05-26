import torch
from hypothesis import given

from tests.assertion import assert_catted_sequence_close, assert_grad_close, assert_packed_sequence_close
from tests.strategy import BATCH_SIZE, device, EMBEDDING_DIM, sizes, TOKEN_SIZE
from torchrua import cat_sequence, pack_sequence, reverse_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
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
    dim=sizes(EMBEDDING_DIM),
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
