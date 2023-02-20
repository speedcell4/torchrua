import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence as excepted_pack_sequence
from torch.nn.utils.rnn import pad_sequence as excepted_pad_sequence

from tests.assertion import assert_close, assert_grad_close
from tests.strategy import sizes, device, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua import pad_sequence, cat_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
)
def test_pad_sequence(data, token_sizes, dim, batch_first):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, _ = pad_sequence(inputs, batch_first=batch_first)
    excepted = excepted_pad_sequence(inputs, batch_first=batch_first)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
)
def test_pad_catted_sequence(data, token_sizes, dim, batch_first):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, actual_token_sizes = pad_sequence(
        cat_sequence(sequences),
        batch_first=batch_first,
    )

    excepted = excepted_pad_sequence(sequences, batch_first=batch_first)
    excepted_token_sizes = torch.tensor(token_sizes, device=device)

    assert_close(actual=actual, expected=excepted)
    assert_close(actual=actual_token_sizes, expected=excepted_token_sizes)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
)
def test_pad_packed_sequence(data, token_sizes, dim, batch_first):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, actual_token_sizes = pad_sequence(
        excepted_pack_sequence(sequences, enforce_sorted=False),
        batch_first=batch_first,
    )

    excepted = excepted_pad_sequence(sequences, batch_first=batch_first)
    excepted_token_sizes = torch.tensor(token_sizes, device=device)

    assert_close(actual=actual, expected=excepted)
    assert_close(actual=actual_token_sizes, expected=excepted_token_sizes)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)
