import torch
from hypothesis import given, settings, strategies as st
from torch.nn.utils.rnn import pad_sequence
from torchnyan import BATCH_SIZE, TOKEN_SIZE, assert_close, device, sizes

from torchrua import Z


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    rua=st.sampled_from(Z.__args__),
    zero_one_dtype=st.sampled_from([
        (False, True, torch.bool),
        (-1, +2, torch.long),
        (torch.finfo(torch.float16).min, torch.finfo(torch.float16).max, torch.float16),
        (torch.finfo(torch.float32).min, torch.finfo(torch.float32).max, torch.float32),
        (torch.finfo(torch.float64).min, torch.finfo(torch.float64).max, torch.float64),
    ])
)
def test_mask(token_sizes, rua, zero_one_dtype):
    inputs = [
        torch.randn((token_size,), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    zero, one, dtype = zero_one_dtype

    actual = rua.new(inputs).mask(zero=zero, one=one, dtype=dtype)
    expected = pad_sequence([
        torch.full((token_size,), fill_value=one, device=device, dtype=dtype)
        for token_size in token_sizes
    ], batch_first=True, padding_value=zero)

    assert_close(actual=actual, expected=expected)
