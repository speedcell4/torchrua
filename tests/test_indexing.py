import torch
from hypothesis import given, strategies as st

from tests.assertions import assert_packed_sequence_close, assert_close, assert_grad_close, assert_catted_sequence_close
from tests.strategies import sizes, device, EMBEDDING_DIM, BATCH_SIZE, TOKEN_SIZE
from torchrua.catting import cat_sequence
from torchrua.indexing import head_catted_sequence, last_catted_sequence, init_sequence, tail_sequence
from torchrua.indexing import head_packed_sequence, last_packed_sequence
from torchrua.packing import pack_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_head_catted_sequence(token_sizes, dim):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = head_catted_sequence(sequence=cat_sequence(sequences))
    expected = torch.stack([sequence[0] for sequence in sequences], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_last_catted_sequence(token_sizes, dim):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = last_catted_sequence(sequence=cat_sequence(sequences))
    expected = torch.stack([sequence[-1] for sequence in sequences], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_init_catted_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    sequences = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = init_sequence(cat_sequence(sequences), n=n)
    expected = cat_sequence([sequence[:-n] for sequence in sequences])

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_tail_catted_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    sequences = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = tail_sequence(cat_sequence(sequences), n=n)
    expected = cat_sequence([sequence[n:] for sequence in sequences])

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    unsort=st.booleans(),
)
def test_head_packed_sequence(token_sizes, dim, unsort):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    packed_sequence = pack_sequence(sequences)
    actual = head_packed_sequence(sequence=packed_sequence, unsort=unsort)
    if not unsort:
        actual = actual[packed_sequence.unsorted_indices]

    expected = torch.stack([sequence[0] for sequence in sequences], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    unsort=st.booleans(),
)
def test_last_packed_sequence(token_sizes, dim, unsort):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    packed_sequence = pack_sequence(sequences)
    actual = last_packed_sequence(sequence=packed_sequence, unsort=unsort)
    if not unsort:
        actual = actual[packed_sequence.unsorted_indices]

    expected = torch.stack([sequence[-1] for sequence in sequences], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_init_packed_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    sequences = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    actual = init_sequence(pack_sequence(sequences), n=n)
    expected = pack_sequence([sequence[:-n] for sequence in sequences])

    assert_packed_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_tail_packed_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    sequences = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = tail_sequence(pack_sequence(sequences), n=n)
    expected = pack_sequence([sequence[n:] for sequence in sequences])

    assert_packed_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=sequences)
