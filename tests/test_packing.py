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
def test_pack_sequence(data, token_sizes, dim, device):
    sequences = [torch.randn((length, dim), device=device) for length in token_sizes]

    pack_tgt = tgt.pack_sequence(sequences, enforce_sorted=False)
    pack_prd = rua.pack_sequence(sequences, device=device)

    assert_close(pack_tgt.data, pack_prd.data)
    assert_equal(pack_tgt.batch_sizes, pack_prd.batch_sizes)
    if pack_tgt.sorted_indices is not None:
        assert_equal(pack_tgt.sorted_indices, pack_prd.sorted_indices)
    if pack_tgt.sorted_indices is not None:
        assert_equal(pack_tgt.unsorted_indices, pack_prd.unsorted_indices)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pack_padded_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [torch.randn((length, dim), device=device) for length in token_sizes]

    pack_tgt = tgt.pack_sequence(sequences, enforce_sorted=False)
    pack_prd = rua.pack_padded_sequence(
        tgt.pad_sequence(sequences, batch_first=batch_first),
        torch.tensor(token_sizes, device=device),
        batch_first=batch_first,
    )

    assert_close(pack_tgt.data, pack_prd.data)
    assert_equal(pack_tgt.batch_sizes, pack_prd.batch_sizes)
    if pack_tgt.sorted_indices is not None:
        assert_equal(pack_tgt.sorted_indices, pack_prd.sorted_indices)
    if pack_tgt.sorted_indices is not None:
        assert_equal(pack_tgt.unsorted_indices, pack_prd.unsorted_indices)
