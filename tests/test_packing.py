import torch
from hypothesis import given, settings, strategies as st
from torch.nn.utils.rnn import pack_sequence
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, assert_grad_close, assert_sequence_close, device, sizes

from torchrua import C, D, P


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua_sequence=st.sampled_from([C.new, D.new, P.new]),
)
def test_pack_sequence(token_sizes, dim, rua_sequence):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua_sequence(inputs).pack()
    expected = pack_sequence(inputs, enforce_sorted=False)

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
