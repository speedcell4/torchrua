import torch
from hypothesis import given, settings, strategies as st
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, assert_close, assert_grad_close, device, sizes

from tests.expected import right_aligned_tensors
from torchrua import Z


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua_sequence=st.sampled_from([z.new for z in Z.__args__]),
)
def test_right_sequence(token_sizes, dim, rua_sequence):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual, _ = rua_sequence(inputs).right(0)
    expected = right_aligned_tensors(inputs, padding_value=0)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)
