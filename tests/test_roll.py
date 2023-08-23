import torch
from hypothesis import given
from hypothesis import strategies as st
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_catted_sequence_close
from torchnyan import assert_grad_close
from torchnyan import device
from torchnyan import sizes

from torchrua import cat_sequence
from torchrua import pack_sequence
from torchrua import pad_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua_sequence=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
)
def test_roll_sequence(data, token_sizes, dim, rua_sequence):
    shifts = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))

    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua_sequence(inputs).roll(shifts=shifts).cat()
    expected = cat_sequence([sequence.roll(shifts, dims=[0]) for sequence in inputs])

    assert_catted_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=inputs)
