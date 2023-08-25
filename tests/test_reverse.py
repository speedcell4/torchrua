import torch
from hypothesis import given
from hypothesis import strategies as st
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_grad_close
from torchnyan import assert_sequence_close
from torchnyan import device
from torchnyan import sizes

from torchrua import C
from torchrua import D
from torchrua import P


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua_sequence=st.sampled_from([C.new, D.new, P.new]),
)
def test_reverse_sequence(token_sizes, dim, rua_sequence):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua_sequence(inputs).rev().cat()
    expected = C.new([sequence.flip(dims=[0]) for sequence in inputs])

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
