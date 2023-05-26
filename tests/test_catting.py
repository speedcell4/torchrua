import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence as torch_pack_sequence, pad_sequence as torch_pad_sequence

from tests.assertion import assert_catted_sequence_close, assert_grad_close
from tests.strategy import BATCH_SIZE, device, EMBEDDING_DIM, sizes, TOKEN_SIZE
from torchrua import cat_packed_sequence, cat_padded_sequence, cat_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_cat_packed_sequence(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = cat_sequence(inputs, device=device)
    expected = cat_packed_sequence(torch_pack_sequence(inputs, enforce_sorted=False), device=device)

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
)
def test_cat_padded_sequence(token_sizes, dim, batch_first):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = cat_sequence(inputs, device=device)
    expected = cat_padded_sequence(
        sequence=torch_pad_sequence(inputs, batch_first=batch_first),
        token_sizes=torch.tensor(token_sizes, device=device),
        batch_first=batch_first,
        device=device,
    )

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
