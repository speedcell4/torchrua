import torch
from hypothesis import given, settings, strategies as st
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, assert_grad_close, assert_sequence_close, device, sizes

from torchrua import C, D, P


@settings(deadline=None)
@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua_sequence=st.sampled_from([C.new, D.new, P.new]),
)
def test_trunc_sequence(data, token_sizes, dim, rua_sequence):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    s = min(token_sizes) - 1
    a = data.draw(st.integers(0, max_value=s))
    b = data.draw(st.integers(0, max_value=s - a))

    actual = rua_sequence(inputs).trunc((a, b)).cat()
    expected = C.new([sequence[a:sequence.size()[0] - b] for sequence in inputs])

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
