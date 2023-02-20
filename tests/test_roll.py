import torch
from hypothesis import given, strategies as st

from tests.assertion import assert_catted_sequence_close, assert_packed_sequence_close, assert_grad_close
from tests.strategy import device, sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua import cat_sequence, pack_sequence, roll_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_roll_catted_sequence(data, token_sizes, dim):
    shifts = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    sequence = cat_sequence(inputs)

    actual = roll_sequence(sequence, shifts=shifts)
    expected = cat_sequence([sequence.roll(shifts, dims=[0]) for sequence in inputs])

    assert_catted_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_roll_packed_sequence(data, token_sizes, dim):
    shifts = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    sequence = pack_sequence(inputs)

    actual = roll_sequence(sequence, shifts=shifts)
    expected = pack_sequence([sequence.roll(shifts, dims=[0]) for sequence in inputs])

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)
