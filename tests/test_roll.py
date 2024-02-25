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
def test_roll_sequence(data, token_sizes, dim, rua_sequence):
    shifts = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua_sequence(inputs).roll(shifts=shifts).cat()
    expected = C.new([sequence.roll(shifts, dims=[0]) for sequence in inputs])

    assert_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)
