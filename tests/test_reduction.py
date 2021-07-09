import torch
from einops import rearrange
from hypothesis import given, strategies as st
from torch import nn

from tests.strategies import batch_size_lists, TINY_BATCH_SIZE, embedding_dims, devices, token_size_lists, \
    TINY_TOKEN_SIZE, TINY_EMBEDDING_DIM
from tests.utils import assert_close, assert_grad_close
from torchrua import cat_sequence, pack_sequence, reduce_catted_sequences


@given(
    data=st.data(),
    batch_sizes=batch_size_lists(max_batch_size=TINY_BATCH_SIZE),
    in_dim=embedding_dims(max_value=TINY_EMBEDDING_DIM),
    hidden_dim=embedding_dims(max_value=TINY_EMBEDDING_DIM),
    device=devices(),
)
def test_reduce_catted_sequences(data, batch_sizes, in_dim, hidden_dim, device):
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

    reduction_pack = reduce_catted_sequences(catted_sequences, device=device)
    _, (prediction, _) = rnn(reduction_pack)
    prediction = rearrange(prediction, 'd n x -> n (d x)')

    target = []
    for pack in packed_sequences:
        _, (t, _) = rnn(pack)
        target.append(rearrange(t, 'd n x -> n (d x)'))
    target = pack_sequence(target).data

    assert_close(prediction, target)
    assert_grad_close(prediction, target, flatten_sequences)
