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
from torchrua import cat_sequence
from torchrua import pack_sequence
from torchrua import pad_sequence


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    rua_sequence=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
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
    expected = cat_sequence([sequence[a:sequence.size()[0] - b] for sequence in inputs])

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
