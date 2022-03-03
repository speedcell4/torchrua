import torch
from einops import rearrange
from hypothesis import given
from torch import nn

from tests.assertions import assert_close, assert_grad_close
from tests.strategies import devices, TINY_TOKEN_SIZE, BATCH_SIZE
from tests.strategies import sizes, EMBEDDING_DIM
from torchrua import cat_sequence, pack_sequence, compose_catted_sequences


@given(
    token_sizes_batch=sizes(BATCH_SIZE, BATCH_SIZE, TINY_TOKEN_SIZE),
    input_size=sizes(EMBEDDING_DIM),
    hidden_size=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_compose_catted_sequences(token_sizes_batch, input_size, hidden_size, device):
    sequences = [
        [
            torch.randn((token_size, input_size), requires_grad=True, device=device)
            for token_size in token_sizes
        ]
        for token_sizes in token_sizes_batch
    ]
    inputs = [token for sequence in sequences for token in sequence]
    catted_sequences = [cat_sequence(sequence, device=device) for sequence in sequences]
    packed_sequences = [pack_sequence(sequence, device=device) for sequence in sequences]

    rnn = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        bidirectional=True, bias=True,
    ).to(device=device)

    _, (actual, _) = rnn(compose_catted_sequences(catted_sequences, device=device))
    actual = rearrange(actual, 'd n x -> n (d x)')

    excepted = []
    for packed_sequence in packed_sequences:
        _, (hidden, _) = rnn(packed_sequence)
        excepted.append(rearrange(hidden, 'd n x -> n (d x)'))
    excepted = pack_sequence(excepted).data

    assert_close(actual, excepted, check_stride=False)
    assert_grad_close(actual, excepted, inputs=inputs)
