from itertools import zip_longest
from typing import List

import torch
from hypothesis import given

from tests.assertion import assert_catted_sequence_close, assert_grad_close, assert_packed_sequence_close
from tests.strategy import BATCH_SIZE, device, EMBEDDING_DIM, sizes, TINY_BATCH_SIZE, TOKEN_SIZE
from torchrua import cat_sequence, cat_sequences, pack_sequence


@given(
    token_sizes_batch=sizes(TINY_BATCH_SIZE, BATCH_SIZE, TOKEN_SIZE),
    embedding_dim=sizes(EMBEDDING_DIM),
)
def test_concat_catted_sequences(token_sizes_batch, embedding_dim):
    sequences_batch = [
        [torch.randn((token_size, embedding_dim), device=device, requires_grad=True) for token_size in token_sizes]
        for token_sizes in token_sizes_batch
    ]

    actual = cat_sequences([
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


@given(
    token_sizes_batch=sizes(TINY_BATCH_SIZE, BATCH_SIZE, TOKEN_SIZE),
    embedding_dim=sizes(EMBEDDING_DIM),
)
def test_concat_packed_sequences(token_sizes_batch: List[List[int]], embedding_dim: int):
    sequences_batch = [
        [torch.randn((token_size, embedding_dim), device=device, requires_grad=True) for token_size in token_sizes]
        for token_sizes in token_sizes_batch
    ]

    actual = cat_sequences([
        pack_sequence(sequences, device=device)
        for sequences in sequences_batch
    ])
    excepted = pack_sequence([
        torch.cat([seq for seq in sequences if seq is not None], dim=0)
        for sequences in zip_longest(*sequences_batch)
    ], device=device)

    assert_packed_sequence_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual.data, expected=excepted.data, inputs=[
        sequence for sequences in sequences_batch for sequence in sequences
    ])
