import torch
from hypothesis import given, strategies as st

from tests.assertions import assert_close, assert_grad_close
from tests.strategies import devices, sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua import pack_sequence, cat_sequence
from torchrua.padding import pad_sequence
from torchrua.reduction import reduce_catted_indices, reduce_sequence
from torchrua.reduction import reduce_packed_indices, reduce_padded_indices


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_reduce_catted_sequence(token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    data, token_sizes = cat_sequence(sequences, device=device)
    indices = reduce_catted_indices(token_sizes=token_sizes)
    actual = reduce_sequence(torch.add)(data, indices)

    excepted, _ = pad_sequence(sequences, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_reduce_packed_sequence(token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    data, batch_sizes, _, unsorted_indices = pack_sequence(sequences, device=device)
    indices = reduce_packed_indices(batch_sizes=batch_sizes, unsorted_indices=unsorted_indices)
    actual = reduce_sequence(torch.add)(data, indices)

    excepted, _ = pad_sequence(sequences, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
    device=devices(),
)
def test_reduce_padded_sequence(token_sizes, dim, batch_first, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    data, token_sizes = pad_sequence(sequences, device=device, batch_first=batch_first)
    indices = reduce_padded_indices(token_sizes=token_sizes, batch_first=batch_first)
    actual = reduce_sequence(torch.add)(data, indices)

    excepted, _ = pad_sequence(sequences, batch_first=True, device=device)
    excepted = excepted.sum(dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=sequences)
