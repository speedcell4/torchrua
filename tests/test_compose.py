import torch
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from torch import nn
from torch.nn.utils.rnn import pack_sequence
from torchnyan import FEATURE_DIM
from torchnyan import TINY_BATCH_SIZE
from torchnyan import TINY_TOKEN_SIZE
from torchnyan import assert_close
from torchnyan import assert_grad_close
from torchnyan import device
from torchnyan import sizes

from torchrua import C
from torchrua import D
from torchrua import P
from torchrua import compose


@given(
    data=st.data(),
    token_sizes_batch=sizes(TINY_BATCH_SIZE, TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
    input_size=sizes(FEATURE_DIM),
    hidden_size=sizes(FEATURE_DIM),
)
@settings(deadline=None)
def test_compose_sequences(data, token_sizes_batch, input_size, hidden_size):
    sequences = [
        [
            torch.randn((token_size, input_size), requires_grad=True, device=device)
            for token_size in token_sizes
        ]
        for token_sizes in token_sizes_batch
    ]

    rnn = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        bidirectional=True, bias=True,
    ).to(device=device)

    actual_sequences = [
        data.draw(st.sampled_from([C, D, P])).new(sequence).to(device=device)
        for sequence in sequences
    ]
    _, (actual, _) = rnn(compose(actual_sequences))
    actual = actual.transpose(-3, -2).flatten(start_dim=-2)

    expected = []
    for sequence in sequences:
        _, (hidden, _) = rnn(pack_sequence(sequence, enforce_sorted=False))
        expected.append(hidden.transpose(-3, -2).flatten(start_dim=-2))
    expected, _, _, _ = pack_sequence(expected, enforce_sorted=False)

    assert_close(actual, expected, check_stride=False)
    assert_grad_close(actual, expected, inputs=[token for sequence in sequences for token in sequence])
