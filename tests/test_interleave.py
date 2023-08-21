from typing import List

import torch
from hypothesis import given
from torch.testing import assert_close
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_grad_close
from torchnyan import device
from torchnyan import sizes

from torchrua.catting import cat_sequence
from torchrua.interleave import repeat_interleave_sequence
from torchrua.packing import pack_sequence
from torchrua.padding import pad_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_repeat_interleave(token_sizes: List[int], dim: int) -> None:
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    repeats = [
        torch.randint(1, 10, (token_size,), device=device)
        for token_size in token_sizes
    ]

    catted_sequence = cat_sequence(inputs, device=device)
    catted_repeats = cat_sequence(repeats, device=device).data

    packed_sequence = pack_sequence(inputs, device=device)
    packed_repeats = pack_sequence(repeats, device=device).data

    catted_sequence = repeat_interleave_sequence(catted_sequence, repeats=catted_repeats)
    packed_sequence = repeat_interleave_sequence(packed_sequence, repeats=packed_repeats)

    catted_data, catted_token_sizes = pad_sequence(catted_sequence, device=device)
    packed_data, packed_token_sizes = pad_sequence(packed_sequence, device=device)

    assert_close(catted_data, packed_data)
    assert_close(catted_token_sizes, packed_token_sizes)
    assert_grad_close(catted_data, packed_data, inputs=inputs)
