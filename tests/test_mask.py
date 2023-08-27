import torch
from hypothesis import given
from hypothesis import strategies as st
from torch.nn.utils.rnn import pad_sequence
from torchnyan import BATCH_SIZE
from torchnyan import TOKEN_SIZE
from torchnyan import assert_close
from torchnyan import device
from torchnyan import sizes

from torchrua import C
from torchrua import D
from torchrua import P


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    rua_sequence=st.sampled_from([C.new, D.new, P.new]),
)
def test_mask_sequence(token_sizes, rua_sequence):
    inputs = [
        torch.randn((token_size,), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    actual = rua_sequence(inputs).mask()
    expected = pad_sequence([
        torch.ones((token_size,), device=device, dtype=torch.bool)
        for token_size in token_sizes
    ], batch_first=True, padding_value=False)

    assert_close(actual=actual, expected=expected)
