import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close, assert_equal
from torchrua import padding as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pad_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]

    pad_tgt = tgt.pad_sequence(sequences, batch_first=batch_first)
    pad_prd = rua.pad_sequence(sequences, batch_first=batch_first)

    assert_close(pad_tgt, pad_prd)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pad_packed_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    packed_sequence = tgt.pack_sequence(sequences, enforce_sorted=False)
    token_sizes_tgt = torch.tensor(token_sizes, device=device)

    pad_tgt = tgt.pad_sequence(sequences, batch_first=batch_first)
    pad_prd, token_sizes_prd = rua.pad_packed_sequence(packed_sequence, batch_first=batch_first)

    assert_close(pad_tgt, pad_prd)
    assert_equal(token_sizes_tgt, token_sizes_prd)
