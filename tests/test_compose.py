import torch
from hypothesis import given
from torch import nn

from tests.assertion import assert_close, assert_grad_close
from tests.strategy import device, EMBEDDING_DIM, sizes, TINY_BATCH_SIZE, TINY_TOKEN_SIZE
from torchrua import cat_sequence, compose_catted_sequences, pack_sequence


@given(
    token_sizes_batch=sizes(TINY_BATCH_SIZE, TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
    input_size=sizes(EMBEDDING_DIM),
    hidden_size=sizes(EMBEDDING_DIM),
)
def test_compose_catted_sequences(token_sizes_batch, input_size, hidden_size):
    sequences = [
        [
            torch.randn((token_size, input_size), requires_grad=True, device=device)
            for token_size in token_sizes
        ]
        for token_sizes in token_sizes_batch
    ]
    inputs = [token for seq in sequences for token in seq]
    catted_sequences = [cat_sequence(seq, device=device) for seq in sequences]
    packed_sequences = [pack_sequence(seq, device=device) for seq in sequences]

    rnn = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        bidirectional=True, bias=True,
    ).to(device=device)

    _, (actual, _) = rnn(compose_catted_sequences(*catted_sequences, device=device))
    actual = actual.transpose(-3, -2).flatten(start_dim=-2)

    excepted = []
    for packed_sequence in packed_sequences:
        _, (hidden, _) = rnn(packed_sequence)
        excepted.append(hidden.transpose(-3, -2).flatten(start_dim=-2))
    excepted = pack_sequence(excepted).data

    assert_close(actual, excepted, check_stride=False)
    assert_grad_close(actual, excepted, inputs=inputs)
