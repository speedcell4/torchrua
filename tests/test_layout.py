import torch
from hypothesis import given, settings, strategies as st
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, assert_grad_close, assert_sequence_close, device, sizes

from tests.expected import install
from torchrua import Z

install()


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    actual_fn=st.sampled_from(Z.__args__),
    expected_fn=st.sampled_from(Z.__args__),
)
def test_cat(token_sizes, dim, actual_fn, expected_fn):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = actual_fn.new(inputs).cat()
    expected = expected_fn.expected_new(inputs, token_sizes).cat()

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    actual_fn=st.sampled_from(Z.__args__),
    expected_fn=st.sampled_from(Z.__args__),
)
def test_left(token_sizes, dim, actual_fn, expected_fn):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = actual_fn.new(inputs).left()
    expected = expected_fn.expected_new(inputs, token_sizes).left()

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    actual_fn=st.sampled_from(Z.__args__),
    expected_fn=st.sampled_from(Z.__args__),
)
def test_pack(token_sizes, dim, actual_fn, expected_fn):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = actual_fn.new(inputs).pack()
    expected = expected_fn.expected_new(inputs, token_sizes).pack()

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    actual_fn=st.sampled_from(Z.__args__),
    expected_fn=st.sampled_from(Z.__args__),
)
def test_right(token_sizes, dim, actual_fn, expected_fn):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = actual_fn.new(inputs).right()
    expected = expected_fn.expected_new(inputs, token_sizes).right()

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
