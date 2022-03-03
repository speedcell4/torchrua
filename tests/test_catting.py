import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence as torch_pack_sequence
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence

from tests.assertions import assert_grad_close, assert_catted_sequence_close
from tests.strategies import devices, sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua.catting import cat_sequence, cat_padded_sequence, cat_packed_sequence, trunc_catted_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_cat_packed_sequence(token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = cat_sequence(sequences, device=device)
    expected = cat_packed_sequence(torch_pack_sequence(sequences, enforce_sorted=False), device=device)

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=sequences)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
    device=devices(),
)
def test_cat_padded_sequence(token_sizes, dim, batch_first, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = cat_sequence(sequences, device=device)
    expected = cat_padded_sequence(
        sequence=torch_pad_sequence(sequences, batch_first=batch_first),
        token_sizes=torch.tensor(token_sizes, device=device),
        batch_first=batch_first,
        device=device,
    )

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_trunc_catted_sequence(data, token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    s = min(token_sizes)
    a = data.draw(st.integers(0, max_value=s))
    b = data.draw(st.integers(0, max_value=s - a))

    actual = trunc_catted_sequence(cat_sequence(sequences, device=device), trunc=(a, b))
    excepted = cat_sequence([sequence[a:sequence.size()[0] - b] for sequence in sequences], device=device)

    assert_catted_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=sequences)
