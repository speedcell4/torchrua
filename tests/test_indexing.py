import torch
from hypothesis import given, strategies as st

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_packed_sequence_close, assert_close, assert_grad_close, assert_equal
from torchrua.catting import cat_sequence
from torchrua.indexing import head_catted_sequence, last_catted_sequence, init_catted_sequence
from torchrua.indexing import head_packed_sequence, last_packed_sequence, init_packed_sequence, tail_packed_sequence
from torchrua.packing import pack_sequence


@given(
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_head_catted_sequence(token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    catted_sequence = cat_sequence(sequences)

    actual = head_catted_sequence(sequence=catted_sequence)
    expected = torch.stack([sequence[0] for sequence in sequences], dim=0)

    assert_close(actual, expected)
    assert_grad_close(actual, expected, inputs=sequences)


@given(
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_last_catted_sequence(token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    catted_sequence = cat_sequence(sequences)

    actual = last_catted_sequence(sequence=catted_sequence)
    expected = torch.stack([sequence[-1] for sequence in sequences], dim=0)

    assert_close(actual, expected)
    assert_grad_close(actual, expected, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_init_catted_sequence(data, token_sizes, dim, device):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    sequences = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    catted_sequence = cat_sequence(sequences)

    actual = init_catted_sequence(sequence=catted_sequence, n=n)
    expected = cat_sequence([sequence[:-n] for sequence in sequences])

    assert_close(actual.data, expected.data)
    assert_equal(actual.token_sizes, expected.token_sizes)
    assert_grad_close(actual.data, expected.data, inputs=sequences)


@given(
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_head_packed_sequence(token_sizes, dim, unsort, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(sequences)

    actual = head_packed_sequence(sequence=packed_sequence, unsort=unsort)
    if not unsort:
        actual = actual[packed_sequence.unsorted_indices]

    expected = torch.stack([sequence[0] for sequence in sequences], dim=0)

    assert_close(actual, expected)
    assert_grad_close(actual, expected, inputs=sequences)


@given(
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_last_packed_sequence(token_sizes, dim, unsort, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(sequences)

    actual = last_packed_sequence(sequence=packed_sequence, unsort=unsort)
    if not unsort:
        actual = actual[packed_sequence.unsorted_indices]

    expected = torch.stack([sequence[-1] for sequence in sequences], dim=0)

    assert_close(actual, expected)
    assert_grad_close(actual, expected, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_init_packed_sequence(data, token_sizes, dim, device):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    sequences = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(sequences)

    actual = init_packed_sequence(sequence=packed_sequence, n=n)
    expected = pack_sequence([sequence[:-n] for sequence in sequences])

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_tail_packed_sequence(data, token_sizes, dim, device):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    sequences = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(sequences)

    actual = tail_packed_sequence(sequence=packed_sequence, n=n)
    expected = pack_sequence([sequence[n:] for sequence in sequences])

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=sequences)
