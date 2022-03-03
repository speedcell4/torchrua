import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close

from tests.assertions import assert_packed_sequence_close, assert_grad_close, assert_equal
from tests.strategies import devices, sizes, EMBEDDING_DIM, BATCH_SIZE, TOKEN_SIZE
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.roll import roll_packed_sequence, roll_catted_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_roll_catted_sequence(data, token_sizes, dim, device):
    offset = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    sequence = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    catted_sequence = cat_sequence(sequence)

    actual = roll_catted_sequence(sequence=catted_sequence, shifts=offset)
    expected = cat_sequence([sequence.roll(offset, dims=[0]) for sequence in sequence])

    assert_close(actual.data, expected.data)
    assert_equal(actual.token_sizes, expected.token_sizes)
    assert_grad_close(actual.data, expected.data, inputs=sequence)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_roll_packed_sequence(data, token_sizes, dim, device):
    offset = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    sequence = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(sequence)

    actual = roll_packed_sequence(sequence=packed_sequence, shifts=offset)
    expected = pack_sequence([sequence.roll(offset, dims=[0]) for sequence in sequence])

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=sequence)
