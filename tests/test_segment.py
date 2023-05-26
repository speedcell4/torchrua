import torch
from hypothesis import given, strategies as st

from tests.assertion import assert_close, assert_grad_close
from tests.strategy import BATCH_SIZE, device, EMBEDDING_DIM, sizes, TOKEN_SIZE
from torchrua import cat_packed_sequence, pack_sequence, segment_sequence
from torchrua.catting import cat_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    reduce=st.sampled_from(['mean', 'sum', 'max', 'min']),
)
def test_segment_sequence(token_sizes, dim, reduce):
    duration = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
        for token_size in token_sizes
    ]

    tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)

    actual = segment_sequence(
        tensor=tensor, reduce=reduce,
        sizes=cat_sequence(sequences=duration, device=device),
    )

    expected = segment_sequence(
        tensor=tensor, reduce=reduce,
        sizes=pack_sequence(sequences=duration, device=device),
    )
    expected = cat_packed_sequence(expected, device=device)

    assert_close(actual=actual.data, expected=expected.data)
    assert_close(actual=actual.token_sizes, expected=expected.token_sizes)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=tensor)
