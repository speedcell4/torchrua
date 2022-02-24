from typing import List

import torch
from hypothesis import given
from torch.testing import assert_close
from torch.types import Device

from tests.strategies import draw_token_sizes, draw_device, draw_embedding_dim
from tests.utils import assert_grad_close
from torchrua.catting import cat_sequence
from torchrua.interleave import repeat_interleave_catted_sequence, repeat_interleave_packed_sequence
from torchrua.packing import pack_sequence
from torchrua.padding import pad_packed_sequence, pad_catted_sequence


@given(
    token_sizes=draw_token_sizes(),
    dim=draw_embedding_dim(),
    device=draw_device(),
)
def test_repeat_interleave(token_sizes: List[int], dim: int, device: Device) -> None:
    sequence = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    repeats = [
        torch.randint(1, 10, (token_size,), device=device)
        for token_size in token_sizes
    ]

    catted_sequence = cat_sequence(sequence, device=device)
    catted_repeats = cat_sequence(repeats, device=device).data

    packed_sequence = pack_sequence(sequence, device=device)
    packed_repeats = pack_sequence(repeats, device=device).data

    catted_sequence = repeat_interleave_catted_sequence(sequence=catted_sequence, repeats=catted_repeats)
    packed_sequence = repeat_interleave_packed_sequence(sequence=packed_sequence, repeats=packed_repeats)

    catted_data, catted_token_sizes = pad_catted_sequence(catted_sequence, batch_first=True, device=device)
    packed_data, packed_token_sizes = pad_packed_sequence(packed_sequence, batch_first=True, device=device)

    assert_close(catted_data, packed_data)
    assert_close(catted_token_sizes, packed_token_sizes)
    assert_grad_close(catted_data, packed_data, inputs=sequence)
