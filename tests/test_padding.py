import torch
from hypothesis import given, strategies as st
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_close
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

    pack_tgt = tgt.pad_sequence(sequences, batch_first=batch_first)
    pack_prd = rua.pad_sequence(sequences, batch_first=batch_first)

    assert_close(pack_tgt, pack_prd)
