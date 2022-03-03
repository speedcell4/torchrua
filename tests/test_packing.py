import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence as torch_pack_sequence
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence

from tests.assertions import assert_packed_sequence_close, assert_grad_close
from tests.strategies import devices, sizes, EMBEDDING_DIM, BATCH_SIZE, TOKEN_SIZE
from torchrua import pack_sequence, trunc_packed_sequence, pack_padded_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_pack_sequence(data, token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = pack_sequence(sequences, device=device)
    excepted = torch_pack_sequence(sequences, enforce_sorted=False)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pack_padded_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = pack_padded_sequence(
        torch_pad_sequence(sequences, batch_first=batch_first),
        torch.tensor(token_sizes, device=device),
        batch_first=batch_first,
    )

    excepted = torch_pack_sequence(sequences, enforce_sorted=False)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_trunc_packed_sequence(data, token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    s = min(token_sizes) - 1  # TODO: support zero-length packing
    a = data.draw(st.integers(0, max_value=s))
    b = data.draw(st.integers(0, max_value=s - a))

    actual = trunc_packed_sequence(pack_sequence(sequences, device=device), trunc=(a, b))
    excepted = pack_sequence([sequence[a:sequence.size()[0] - b] for sequence in sequences], device=device)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=sequences)
