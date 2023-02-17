import torch
from hypothesis import given, strategies as st

from tests.assertion import assert_catted_sequence_close, assert_packed_sequence_close, assert_close, assert_grad_close
from tests.strategy import sizes, device, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua.catting import cat_sequence
from torchrua.indexing import head_catted_sequence, last_catted_sequence
from torchrua.indexing import head_packed_sequence, last_packed_sequence
from torchrua.indexing import init_sequence, tail_sequence
from torchrua.packing import pack_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_head_catted_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = head_catted_sequence(sequence=cat_sequence(inputs))
    expected = torch.stack([sequence[0] for sequence in inputs], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_last_catted_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = last_catted_sequence(sequence=cat_sequence(inputs))
    expected = torch.stack([sequence[-1] for sequence in inputs], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_init_catted_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    inputs = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = init_sequence(cat_sequence(inputs), n=n)
    expected = cat_sequence([sequence[:-n] for sequence in inputs])

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_tail_catted_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    inputs = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = tail_sequence(cat_sequence(inputs), n=n)
    expected = cat_sequence([sequence[n:] for sequence in inputs])

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    unsort=st.booleans(),
)
def test_head_packed_sequence(token_sizes, dim, unsort):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    sequence = pack_sequence(inputs)
    actual = head_packed_sequence(sequence=sequence, unsort=unsort)
    if not unsort:
        actual = actual[sequence.unsorted_indices]

    expected = torch.stack([sequence[0] for sequence in inputs], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    unsort=st.booleans(),
)
def test_last_packed_sequence(token_sizes, dim, unsort):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    sequence = pack_sequence(inputs)
    actual = last_packed_sequence(sequence=sequence, unsort=unsort)
    if not unsort:
        actual = actual[sequence.unsorted_indices]

    expected = torch.stack([sequence[-1] for sequence in inputs], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_init_packed_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    inputs = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    actual = init_sequence(pack_sequence(inputs), n=n)
    expected = pack_sequence([sequence[:-n] for sequence in inputs])

    assert_packed_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_tail_packed_sequence(data, token_sizes, dim):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    inputs = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = tail_sequence(pack_sequence(inputs), n=n)
    expected = pack_sequence([sequence[n:] for sequence in inputs])

    assert_packed_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
