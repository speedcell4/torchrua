import torch
from hypothesis import given, strategies as st

from tests.assertion import assert_close, assert_grad_close
from tests.strategy import BATCH_SIZE, device, EMBEDDING_DIM, sizes, TOKEN_SIZE
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.padding import pad_sequence
from torchrua.reduction import reduce_catted_sequence, reduce_packed_sequence, reduce_padded_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_reduce_catted_sequence(token_sizes, dim):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    catted_sequence = cat_sequence(sequences, device=device)
    actual = reduce_catted_sequence(torch.add)(catted_sequence)

    excepted, _ = pad_sequence(sequences, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_reduce_packed_sequence(token_sizes, dim):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    packed_sequence = pack_sequence(sequences, device=device)
    actual = reduce_packed_sequence(torch.add)(packed_sequence)

    excepted, _ = pad_sequence(sequences, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
)
def test_reduce_padded_sequence(token_sizes, dim, batch_first):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    padded_sequence = pad_sequence(sequences, batch_first=batch_first, device=device)
    actual = reduce_padded_sequence(torch.add)(padded_sequence, batch_first=batch_first)

    excepted, _ = pad_sequence(sequences, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)
