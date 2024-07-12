import torch
from hypothesis import given, settings, strategies as st
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, device, sizes
from torchnyan.assertion import assert_close

from torchrua import Z


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua=st.sampled_from(Z.__args__),
)
def test_split(token_sizes, dim, rua):
    inputs = expected = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua.new(inputs).split()

    for a, e in zip(actual, expected):
        assert_close(actual=a, expected=e)
