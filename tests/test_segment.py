import torch
from hypothesis import given
from torch.testing import assert_close

from tests.assertions import assert_equal
from tests.strategies import devices, sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua import cat_sequence, pack_sequence, cat_packed_sequence, segment_sequence


@given(
    device=devices(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
)
def test_segment_sequence(device, token_sizes, dim):
    chunk_sizes = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
        for token_size in token_sizes
    ]

    tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)

    catted_sizes = cat_sequence(sequences=chunk_sizes, device=device)
    packed_sizes = pack_sequence(sequences=chunk_sizes, device=device)

    actual = segment_sequence(catted_sizes, tensor=tensor, reduce='max', batch_first=True)

    expected = segment_sequence(packed_sizes, tensor=tensor, reduce='max', batch_first=True)
    expected = cat_packed_sequence(expected, device=device)

    assert_close(actual=actual.data, expected=expected.data)
    assert_equal(actual=actual.token_sizes, expected=expected.token_sizes)
