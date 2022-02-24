import torch
from einops import rearrange
from hypothesis import given, strategies as st
from torch import nn

from tests.strategies import TINY_BATCH_SIZE, TINY_TOKEN_SIZE, TINY_EMBEDDING_DIM
from tests.strategies import draw_embedding_dim, draw_device, draw_token_sizes, draw_batch_sizes
from tests.utils import assert_close, assert_grad_close
from torchrua import cat_sequence, pack_sequence, reduce_catted_sequences


@given(
    data=st.data(),
    batch_sizes=draw_batch_sizes(max_batch_size=TINY_BATCH_SIZE),
    in_dim=draw_embedding_dim(max_value=TINY_EMBEDDING_DIM),
    hidden_dim=draw_embedding_dim(max_value=TINY_EMBEDDING_DIM),
    device=draw_device(),
)
def test_reduce_catted_sequences(data, batch_sizes, in_dim, hidden_dim, device):
    sequences = [
        [
            torch.randn((token_size, in_dim), requires_grad=True, device=device)
            for token_size in data.draw(
            draw_token_sizes(max_token_size=TINY_TOKEN_SIZE, max_batch_size=TINY_BATCH_SIZE))
        ]
        for _ in batch_sizes
    ]
    inputs = [token for sequence in sequences for token in sequence]
    catted_sequences = [cat_sequence(sequence, device=device) for sequence in sequences]
    packed_sequences = [pack_sequence(sequence, device=device) for sequence in sequences]

    rnn = nn.LSTM(
        input_size=in_dim,
        hidden_size=hidden_dim,
        bidirectional=True, bias=True,
    ).to(device=device)

    reduction_pack = reduce_catted_sequences(catted_sequences, device=device)
    _, (actual, _) = rnn(reduction_pack)
    actual = rearrange(actual, 'd n x -> n (d x)')

    excepted = []
    for pack in packed_sequences:
        _, (t, _) = rnn(pack)
        excepted.append(rearrange(t, 'd n x -> n (d x)'))
    excepted = pack_sequence(excepted).data

    assert_close(actual, excepted, check_stride=False)
    assert_grad_close(actual, excepted, inputs=inputs)
