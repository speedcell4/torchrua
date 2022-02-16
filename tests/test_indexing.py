import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_packed_sequence_close, assert_close, assert_grad_close
from torchrua.roll import roll_packed_sequence
from torchrua.indexing import select_head, select_last, select_init, select_tail


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_select_head(data, token_sizes, dim, unsort, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_pack_sequence = pack_sequence(inputs, enforce_sorted=False)

    actual = select_head(sequence=packed_pack_sequence, unsort=unsort)
    if not unsort:
        actual = actual[packed_pack_sequence.unsorted_indices]

    expected = torch.stack([sequence[0] for sequence in inputs], dim=0)

    assert_close(actual, expected)
    assert_grad_close(actual, expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_select_last(data, token_sizes, dim, unsort, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(inputs, enforce_sorted=False)

    actual = select_last(sequence=packed_sequence, unsort=unsort)
    if not unsort:
        actual = actual[packed_sequence.unsorted_indices]

    expected = torch.stack([sequence[-1] for sequence in inputs], dim=0)

    assert_close(actual, expected)
    assert_grad_close(actual, expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_select_init(data, token_sizes, dim, device):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    inputs = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(inputs, enforce_sorted=False)

    actual = select_init(sequence=packed_sequence, n=n)
    expected = pack_sequence([sequence[:-n] for sequence in inputs], enforce_sorted=False)

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_select_tail(data, token_sizes, dim, device):
    n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))

    inputs = [
        torch.randn((token_size + 1, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(inputs, enforce_sorted=False)

    actual = select_tail(sequence=packed_sequence, n=n)
    expected = pack_sequence([sequence[n:] for sequence in inputs], enforce_sorted=False)

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_roll_packed_sequence(data, token_sizes, dim, device):
    offset = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(inputs, enforce_sorted=False)

    actual = roll_packed_sequence(sequence=packed_sequence, shifts=offset)
    expected = pack_sequence([
        sequence.roll(offset, dims=[0]) for sequence in inputs], enforce_sorted=False)

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)
