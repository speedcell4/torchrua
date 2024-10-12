import torch
from hypothesis import given, settings, strategies as st
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, device, sizes
from torchnyan.assertion import assert_close, assert_grad_close, assert_sequence_close

from torchrua import C, Z


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua=st.sampled_from(Z.__args__),
)
def test_head(token_sizes, dim, rua):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua.new(inputs).head()
    expected = torch.stack([tensor[0] for tensor in inputs], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua=st.sampled_from(Z.__args__),
)
def test_last(token_sizes, dim, rua):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua.new(inputs).last()
    expected = torch.stack([tensor[-1] for tensor in inputs], dim=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua=st.sampled_from(Z.__args__),
)
def test_rev(token_sizes, dim, rua):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua.new(inputs).rev().cat()
    expected = C.new([tensor.flip(dims=[0]) for tensor in inputs])

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@settings(deadline=None)
@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua=st.sampled_from(Z.__args__),
)
def test_roll(data, token_sizes, dim, rua):
    shifts = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua.new(inputs).roll(shifts=shifts).cat()
    expected = C.new([tensor.roll(shifts, dims=[0]) for tensor in inputs])

    assert_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)


@settings(deadline=None)
@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua=st.sampled_from(Z.__args__),
)
def test_trunc(data, token_sizes, dim, rua):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    s = min(token_sizes) - 1
    a = data.draw(st.integers(0, max_value=s))
    b = data.draw(st.integers(0, max_value=s - a))

    actual = rua.new(inputs).trunc((a, b)).cat()
    expected = C.new([tensor[a:tensor.size()[0] - b] for tensor in inputs])

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
