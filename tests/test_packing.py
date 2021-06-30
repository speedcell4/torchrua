import torch
from einops import rearrange
from hypothesis import given, strategies as st
from torch import nn
from torch.nn.utils import rnn as tgt
from torchrua.packing import pack_sequence

from torchrua.catting import cat_sequence

from tests.strategies import TINY_BATCH_SIZE, TINY_TOKEN_SIZE
from tests.strategies import token_size_lists, embedding_dims, devices, batch_size_lists
from tests.utils import assert_packed_close, assert_packed_grad_close, assert_close, assert_grad_close
from torchrua import packing as rua


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_pack_sequence(data, token_sizes, dim, device):
    sequences = [torch.randn((token_size, dim), device=device, requires_grad=True) for token_size in token_sizes]

    target = tgt.pack_sequence(sequences, enforce_sorted=False)
    prediction = rua.pack_sequence(sequences, device=device)

    assert_packed_close(prediction, target)
    assert_packed_grad_close(prediction, target, inputs=sequences)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    batch_first=st.booleans(),
    device=devices(),
)
def test_pack_padded_sequence(data, token_sizes, dim, batch_first, device):
    sequences = [torch.randn((token_size, dim), device=device, requires_grad=True) for token_size in token_sizes]
    padded_sequence = tgt.pad_sequence(sequences, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)

    prediction = rua.pack_padded_sequence(padded_sequence, token_sizes, batch_first=batch_first)
    target = tgt.pack_sequence(sequences, enforce_sorted=False)

    assert_packed_close(prediction, target)
    assert_packed_grad_close(prediction, target, inputs=sequences)


@given(
    data=st.data(),
    batch_sizes=batch_size_lists(max_batch_size=TINY_BATCH_SIZE),
    in_dim=embedding_dims(),
    hidden_dim=embedding_dims(),
    device=devices(),
)
def test_pack_catted_sequences(data, batch_sizes, in_dim, hidden_dim, device):
    sequences = [
        [
            torch.randn((token_size, in_dim), requires_grad=True, device=device)
            for token_size in data.draw(
            token_size_lists(max_token_size=TINY_TOKEN_SIZE, max_batch_size=TINY_BATCH_SIZE))
        ]
        for _ in batch_sizes
    ]
    flatten_sequences = [token for sequence in sequences for token in sequence]
    catted_sequences = [cat_sequence(sequence, device=device) for sequence in sequences]
    packed_sequences = [pack_sequence(sequence, device=device) for sequence in sequences]

    rnn = nn.LSTM(
        input_size=in_dim,
        hidden_size=hidden_dim,
        bidirectional=True, bias=True,
    ).to(device=device)

    reduction_pack = rua.pack_catted_sequences(catted_sequences, device=device)
    _, (prediction, _) = rnn(reduction_pack)
    prediction = rearrange(prediction, 'd n x -> n (d x)')

    target = []
    for pack in packed_sequences:
        _, (t, _) = rnn(pack)
        target.append(rearrange(t, 'd n x -> n (d x)'))
    target = pack_sequence(target).data

    assert_close(prediction, target)
    assert_grad_close(prediction, target, flatten_sequences)
