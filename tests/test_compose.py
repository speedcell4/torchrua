import torch
from hypothesis import given, settings, strategies as st
from torch import nn
from torch.nn.utils.rnn import pack_sequence
from torchnyan import FEATURE_DIM, TINY_BATCH_SIZE, TINY_TOKEN_SIZE, assert_close, assert_grad_close, device, sizes

from torchrua import Z, compose


@settings(deadline=None)
@given(
    data=st.data(),
    token_sizes_batch=sizes(TINY_BATCH_SIZE, TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
    input_size=sizes(FEATURE_DIM),
    hidden_size=sizes(FEATURE_DIM),
)
def test_compose(data, token_sizes_batch, input_size, hidden_size):
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

    actual = [
        data.draw(st.sampled_from(Z.__args__)).new(sequence).to(device=device)
        for sequence in sequences
    ]
    _, (actual, _) = rnn(compose(actual))
    actual = actual.transpose(-3, -2).flatten(start_dim=-2)

    expected = []
    for sequence in sequences:
        _, (hidden, _) = rnn(pack_sequence(sequence, enforce_sorted=False))
        expected.append(hidden.transpose(-3, -2).flatten(start_dim=-2))
    expected, _, _, _ = pack_sequence(expected, enforce_sorted=False)

    assert_close(actual, expected, check_stride=False)
    assert_grad_close(actual, expected, inputs=[token for sequence in sequences for token in sequence])
