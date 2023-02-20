import torch
from hypothesis import given, strategies as st

from tests.assertion import assert_close, assert_grad_close
from tests.strategy import device, sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua import cat_sequence, pack_sequence, cat_packed_sequence, segment_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    reduce=st.sampled_from(['mean', 'sum', 'max', 'min']),
    batch_first=st.booleans(),
)
def test_segment_sequence(token_sizes, dim, reduce, batch_first):
    chunk_sizes = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
        for token_size in token_sizes
    ]

    if batch_first:
        tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)
    else:
        tensor = torch.randn((max(token_sizes), len(token_sizes), dim), device=device, requires_grad=True)

    actual = segment_sequence(
        cat_sequence(sequences=chunk_sizes, device=device),
        tensor=tensor, reduce=reduce, batch_first=batch_first,
    )

    expected = segment_sequence(
        pack_sequence(sequences=chunk_sizes, device=device),
        tensor=tensor, reduce=reduce, batch_first=batch_first,
    )
    expected = cat_packed_sequence(expected, device=device)

    assert_close(actual=actual.data, expected=expected.data)
    assert_close(actual=actual.token_sizes, expected=expected.token_sizes)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=tensor)
