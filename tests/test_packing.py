import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_packed_close
from torchrua import packing as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_pack_sequence(data, token_sizes, dim, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]

    target = tgt.pack_sequence(sequences, enforce_sorted=False)
    prediction = rua.pack_sequence(sequences, device=device)

    assert_packed_close(target, prediction)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pack_padded_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    padded_sequence = tgt.pad_sequence(sequences, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)

    pack_target = tgt.pack_sequence(sequences, enforce_sorted=False)
    pack_prediction = rua.pack_padded_sequence(padded_sequence, token_sizes, batch_first=batch_first)

    assert_packed_close(pack_target, pack_prediction)
