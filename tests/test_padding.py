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

    target = tgt.pad_sequence(sequences, batch_first=batch_first)
    prediction = rua.pad_sequence(sequences, batch_first=batch_first)

    assert_close(target, prediction)


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
    token_sizes_target = torch.tensor(token_sizes, device=torch.device('cpu'))

    pad_target = tgt.pad_sequence(sequences, batch_first=batch_first)
    pad_prediction, token_sizes_prediction = rua.pad_packed_sequence(packed_sequence, batch_first=batch_first)

    assert_close(pad_target, pad_prediction)
    assert_equal(token_sizes_target, token_sizes_prediction)
