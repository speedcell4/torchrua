import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close, assert_equal
from torchrua import packing as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_cat_packed_sequence(data, token_sizes, dim, device):
    sequences = [torch.randn((length, dim), device=device) for length in token_sizes]

    print(f'token_sizes => {token_sizes}')

    pack_tgt = tgt.pack_sequence(sequences, enforce_sorted=False)
    pack_prd = rua.pack_sequence(sequences, device=device)

    assert_close(pack_tgt.data, pack_prd.data)
    assert_equal(pack_tgt.batch_sizes, pack_prd.batch_sizes)
    if pack_tgt.sorted_indices is not None:
        assert_equal(pack_tgt.sorted_indices, pack_prd.sorted_indices)
    if pack_tgt.sorted_indices is not None:
        assert_equal(pack_tgt.unsorted_indices, pack_prd.unsorted_indices)
