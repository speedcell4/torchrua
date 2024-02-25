import torch
from hypothesis import given, settings, strategies as st
from torch.nn.utils.rnn import pad_sequence
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, assert_close, assert_grad_close, device, sizes

from torchrua import C, D, P


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua_sequence=st.sampled_from([C.new, D.new, P.new]),
)
def test_pad_sequence(token_sizes, dim, rua_sequence):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, _ = rua_sequence(inputs).pad(0)
    expected = pad_sequence(inputs, batch_first=True, padding_value=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)
