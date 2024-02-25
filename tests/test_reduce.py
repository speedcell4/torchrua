import torch
from hypothesis import given, settings, strategies as st
from torchnyan import BATCH_SIZE, FEATURE_DIM, TINY_TOKEN_SIZE, assert_close, assert_grad_close, device, sizes

from torchrua import C, scatter_logsumexp, scatter_max, scatter_mean, scatter_min, scatter_prod, scatter_sum, \
    segment_head, segment_last, segment_logsumexp, segment_max, segment_mean, segment_min, segment_prod, segment_sum


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_max(token_sizes, dim, include_self):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    index = [
        torch.full((token_size,), fill_value=index, dtype=torch.long, device=device)
        for index, token_size in enumerate(token_sizes)
    ]
    index = torch.cat(index, dim=0)
    permutation = torch.randperm(sum(token_sizes), dtype=torch.long, device=device)

    tensor = torch.randn((len(token_sizes), dim), dtype=torch.float32, device=device)
    expected = torch.cat([torch.max(seq, dim=0, keepdim=True).values for seq in inputs], dim=0)
    if include_self:
        expected = torch.maximum(expected, tensor)

    source, _ = C.new(inputs)
    actual = scatter_max(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_min(token_sizes, dim, include_self):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    index = [
        torch.full((token_size,), fill_value=index, dtype=torch.long, device=device)
        for index, token_size in enumerate(token_sizes)
    ]
    index = torch.cat(index, dim=0)
    permutation = torch.randperm(sum(token_sizes), dtype=torch.long, device=device)

    tensor = torch.randn((len(token_sizes), dim), dtype=torch.float32, device=device)
    expected = torch.cat([torch.min(seq, dim=0, keepdim=True).values for seq in inputs], dim=0)
    if include_self:
        expected = torch.minimum(expected, tensor)

    source, _ = C.new(inputs)
    actual = scatter_min(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_sum(token_sizes, dim, include_self):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    index = [
        torch.full((token_size,), fill_value=index, dtype=torch.long, device=device)
        for index, token_size in enumerate(token_sizes)
    ]
    index = torch.cat(index, dim=0)
    permutation = torch.randperm(sum(token_sizes), dtype=torch.long, device=device)

    tensor = torch.randn((len(token_sizes), dim), dtype=torch.float32, device=device)
    expected = torch.cat([torch.sum(seq, dim=0, keepdim=True) for seq in inputs], dim=0)
    if include_self:
        expected = expected + tensor

    source, _ = C.new(inputs)
    actual = scatter_sum(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_mean(token_sizes, dim, include_self):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    index = [
        torch.full((token_size,), fill_value=index, dtype=torch.long, device=device)
        for index, token_size in enumerate(token_sizes)
    ]
    index = torch.cat(index, dim=0)
    permutation = torch.randperm(sum(token_sizes), dtype=torch.long, device=device)

    token_sizes = torch.tensor(token_sizes, dtype=torch.float32, device=device)

    tensor = torch.randn((len(token_sizes), dim), dtype=torch.float32, device=device)
    expected = torch.cat([torch.sum(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    if not include_self:
        expected = expected / token_sizes[:, None]
    else:
        expected = (expected + tensor) / (token_sizes[:, None] + 1)

    source, _ = C.new(inputs)
    actual = scatter_mean(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_prod(token_sizes, dim, include_self):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    index = [
        torch.full((token_size,), fill_value=index, dtype=torch.long, device=device)
        for index, token_size in enumerate(token_sizes)
    ]
    index = torch.cat(index, dim=0)
    permutation = torch.randperm(sum(token_sizes), dtype=torch.long, device=device)

    tensor = torch.randn((len(token_sizes), dim), dtype=torch.float32, device=device)
    expected = torch.cat([torch.prod(seq, dim=0, keepdim=True) for seq in inputs], dim=0)
    if include_self:
        expected = expected * tensor

    source, _ = C.new(inputs)
    actual = scatter_prod(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_logsumexp(token_sizes, dim, include_self):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    index = [
        torch.full((token_size,), fill_value=index, dtype=torch.long, device=device)
        for index, token_size in enumerate(token_sizes)
    ]
    index = torch.cat(index, dim=0)
    permutation = torch.randperm(sum(token_sizes), dtype=torch.long, device=device)

    tensor = torch.randn((len(token_sizes), dim), dtype=torch.float32, device=device)
    expected = torch.cat([torch.logsumexp(seq, dim=0, keepdim=True) for seq in inputs], dim=0)
    if include_self:
        expected = torch.logaddexp(expected, tensor)

    source, _ = C.new(inputs)
    actual = scatter_logsumexp(
        tensor, include_self=include_self,
        index=index[permutation], source=source[permutation],
    )

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_max(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.max(seq, dim=0, keepdim=True).values for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_max(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_min(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.min(seq, dim=0, keepdim=True).values for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_min(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_sum(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.sum(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_sum(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_mean(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.mean(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_mean(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_prod(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.prod(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_prod(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_logsumexp(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.logsumexp(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_logsumexp(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_head(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([seq[:1] for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_head(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TINY_TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_last(token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([seq[-1:] for seq in inputs], dim=0)

    tensor, segment_sizes = C.new(inputs)
    actual = segment_last(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)
