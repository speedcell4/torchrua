import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence as excepted_pack_sequence
from torch.nn.utils.rnn import pad_sequence as excepted_pad_sequence

from tests.assertion import assert_packed_sequence_close, assert_grad_close
from tests.strategy import sizes, device, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua import pack_sequence, pack_padded_sequence, cat_sequence, pack_catted_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_pack_sequence(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = pack_sequence(inputs, device=device)
    excepted = excepted_pack_sequence(inputs, enforce_sorted=False)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_pack_catted_sequence(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = pack_catted_sequence(cat_sequence(inputs))

    excepted = excepted_pack_sequence(inputs, enforce_sorted=False)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
)
def test_pack_padded_sequence(data, token_sizes, dim, batch_first):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = pack_padded_sequence(
        excepted_pad_sequence(inputs, batch_first=batch_first),
        torch.tensor(token_sizes, device=device),
        batch_first=batch_first,
    )

    excepted = excepted_pack_sequence(inputs, enforce_sorted=False)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=inputs)
