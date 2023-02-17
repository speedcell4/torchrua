import torch
from hypothesis import given, strategies as st

from tests.assertion import assert_catted_sequence_close, assert_packed_sequence_close, assert_grad_close
from tests.strategy import sizes, device, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua import cat_sequence, pack_sequence, trunc_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_trunc_catted_sequence(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    s = min(token_sizes)
    a = data.draw(st.integers(0, max_value=s))
    b = data.draw(st.integers(0, max_value=s - a))

    actual = trunc_sequence(cat_sequence(inputs, device=device), trunc=(a, b))
    excepted = cat_sequence([sequence[a:sequence.size()[0] - b] for sequence in inputs], device=device)

    assert_catted_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_trunc_packed_sequence(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    s = min(token_sizes) - 1  # TODO: support zero-length packing
    a = data.draw(st.integers(0, max_value=s))
    b = data.draw(st.integers(0, max_value=s - a))

    actual = trunc_sequence(pack_sequence(inputs, device=device), trunc=(a, b))
    excepted = pack_sequence([sequence[a:sequence.size()[0] - b] for sequence in inputs], device=device)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=inputs)
