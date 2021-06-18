import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close, assert_equal
from torchrua import catting as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_cat_packed_sequence(data, token_sizes, dim, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    packed_sequence = tgt.pack_sequence(sequences, enforce_sorted=False)

    data_tgt, token_sizes_tgt = rua.cat_sequence(sequences, device=device)
    data_prd, token_sizes_prd = rua.cat_packed_sequence(packed_sequence, device=device)

    assert_close(data_tgt, data_prd)
    assert_equal(token_sizes_tgt, token_sizes_prd)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_cat_padded_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    padded_sequence = tgt.pad_sequence(sequences, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)

    data_tgt, token_sizes_tgt = rua.cat_sequence(sequences, device=device)
    data_prd, token_sizes_prd = rua.cat_padded_sequence(
        padded_sequence, token_sizes, batch_first=batch_first, device=device)

    assert_close(data_tgt, data_prd)
    assert_equal(token_sizes_tgt, token_sizes_prd)
