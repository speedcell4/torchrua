import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close, assert_equal
from torchrua import catting as rua


@given(
    data=st.data(),
    lengths=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_cat_packed_sequence(data, lengths, dim, device):
    sequences = [torch.randn((length, dim), device=device).mul(10).long() for length in lengths]

    data_tgt, lengths_tgt = rua.cat_sequence(sequences, device=device)
    data_prd, lengths_prd = rua.cat_packed_sequence(
        tgt.pack_sequence(sequences, enforce_sorted=False), device=device)

    assert_close(data_tgt, data_prd)
    assert_equal(lengths_tgt, lengths_prd)
