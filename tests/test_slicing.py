import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from tests.strategies import token_size_lists, embedding_dims, devices, batch_sizes, TINY_BATCH_SIZE
from tests.utils import assert_equal, assert_close
from torchrua.joining import stack_packed_sequences
from torchrua.slicing import chunk_packed_sequence


@given(
    batch_size=batch_sizes(max_value=TINY_BATCH_SIZE),
    token_sizes=token_size_lists(),
    embedding_dim=embedding_dims(),
    dim=st.sampled_from([0, 1]),
    batch_first=st.booleans(),
    device=devices(),
)
def test_chunk_packed_sequence(batch_size, token_sizes, embedding_dim, dim, batch_first, device):
    target = sequences = [
        pack_sequence([
            torch.randn((token_size, embedding_dim), device=device, requires_grad=True)
            for token_size in token_sizes
        ], enforce_sorted=False)
        for _ in range(batch_size)
    ]
    prediction = chunk_packed_sequence(
        sequence=stack_packed_sequences(sequences=sequences, dim=dim),
        chunks=len(sequences), dim=dim,
    )

    for t, p in zip(target, prediction):
        data_prediction, token_sizes_prediction = pad_packed_sequence(p, batch_first=batch_first)
        data_target, token_sizes_target = pad_packed_sequence(t, batch_first=batch_first)

        assert_close(data_prediction, data_target)
        assert_equal(token_sizes_prediction, token_sizes_target)
