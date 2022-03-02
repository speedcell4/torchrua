from itertools import zip_longest
from typing import List

import torch
from hypothesis import given, strategies as st
from torch.types import Device

from tests.strategies import draw_device, draw_token_sizes, MAX_BATCH_SIZE, draw_embedding_dim
from tests.utils import assert_catted_sequence_close, assert_grad_close
from torchrua import cat_sequence, cat_catted_sequences


@given(
    device=draw_device(),
    token_sizes_batch=st.lists(draw_token_sizes(), min_size=1, max_size=MAX_BATCH_SIZE),
    embedding_dim=draw_embedding_dim(),
)
def test_cat_catted_sequences(token_sizes_batch: List[List[int]], embedding_dim: int, device: Device):
    sequences_batch = [
        [torch.randn((token_size, embedding_dim), device=device, requires_grad=True) for token_size in token_sizes]
        for token_sizes in token_sizes_batch
    ]

    actual = cat_catted_sequences([
        cat_sequence(sequences, device=device)
        for sequences in sequences_batch
    ])
    excepted = cat_sequence([
        torch.cat([seq for seq in sequences if seq is not None], dim=0)
        for sequences in zip_longest(*sequences_batch)
    ], device=device)

    assert_catted_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=[
        sequence for sequences in sequences_batch for sequence in sequences
    ])
