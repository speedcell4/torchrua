import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from tests.assertion import assert_close
from tests.strategy import sizes, device, TINY_BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM, BATCH_SIZE
from torchrua.chunk import chunk_packed_sequence
from torchrua.joining import stack_packed_sequences


@given(
    batch_size=sizes(TINY_BATCH_SIZE),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    embedding_dim=sizes(EMBEDDING_DIM),
    dim=sizes(1),
    batch_first=st.booleans(),
)
def test_chunk_packed_sequence(batch_size, token_sizes, embedding_dim, dim, batch_first):
    excepted_sequences = sequences = [
        pack_sequence([
            torch.randn((token_size, embedding_dim), device=device, requires_grad=True)
            for token_size in token_sizes
        ], enforce_sorted=False)
        for _ in range(batch_size)
    ]
    actual_sequences = chunk_packed_sequence(
        sequence=stack_packed_sequences(sequences=sequences, dim=dim),
        chunks=len(sequences), dim=dim,
    )

    for actual_sequence, excepted_sequence in zip(actual_sequences, excepted_sequences):
        actual, actual_token_sizes = pad_packed_sequence(actual_sequence, batch_first=batch_first)
        excepted, excepted_token_sizes = pad_packed_sequence(excepted_sequence, batch_first=batch_first)

        assert_close(actual, excepted)
        assert_close(actual_token_sizes, excepted_token_sizes)
