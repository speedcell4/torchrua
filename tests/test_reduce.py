import torch
from hypothesis import given
from hypothesis import strategies as st
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_close
from torchnyan import assert_grad_close
from torchnyan import device
from torchnyan import sizes

from torchrua import cat_sequence
from torchrua import scatter_logsumexp
from torchrua import scatter_max
from torchrua import scatter_mean
from torchrua import scatter_min
from torchrua import scatter_prod
from torchrua import scatter_sum
from torchrua import segment_logsumexp
from torchrua import segment_max
from torchrua import segment_mean
from torchrua import segment_min
from torchrua import segment_prod
from torchrua import segment_sum


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_max(data, token_sizes, dim, include_self):
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

    source, _ = cat_sequence(inputs, device=device)
    actual = scatter_max(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_min(data, token_sizes, dim, include_self):
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

    source, _ = cat_sequence(inputs, device=device)
    actual = scatter_min(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_sum(data, token_sizes, dim, include_self):
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

    source, _ = cat_sequence(inputs, device=device)
    actual = scatter_sum(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_mean(data, token_sizes, dim, include_self):
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

    source, _ = cat_sequence(inputs, device=device)
    actual = scatter_mean(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_prod(data, token_sizes, dim, include_self):
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

    source, _ = cat_sequence(inputs, device=device)
    actual = scatter_prod(tensor, index=index[permutation], source=source[permutation], include_self=include_self)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    include_self=st.booleans(),
)
def test_scatter_logsumexp(data, token_sizes, dim, include_self):
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

    source, _ = cat_sequence(inputs, device=device)
    actual = scatter_logsumexp(
        tensor, include_self=include_self,
        index=index[permutation], source=source[permutation],
    )

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_max(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.max(seq, dim=0, keepdim=True).values for seq in inputs], dim=0)

    tensor, segment_sizes = cat_sequence(inputs, device=device)
    actual = segment_max(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_min(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.min(seq, dim=0, keepdim=True).values for seq in inputs], dim=0)

    tensor, segment_sizes = cat_sequence(inputs, device=device)
    actual = segment_min(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_sum(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.sum(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = cat_sequence(inputs, device=device)
    actual = segment_sum(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_mean(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.mean(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = cat_sequence(inputs, device=device)
    actual = segment_mean(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_prod(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.prod(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = cat_sequence(inputs, device=device)
    actual = segment_prod(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)


@given(
    data=st.data(),
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
)
def test_segment_logsumexp(data, token_sizes, dim):
    inputs = [
        torch.randn((token_size, dim), dtype=torch.float32, device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    expected = torch.cat([torch.logsumexp(seq, dim=0, keepdim=True) for seq in inputs], dim=0)

    tensor, segment_sizes = cat_sequence(inputs, device=device)
    actual = segment_logsumexp(tensor, segment_sizes=segment_sizes)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=inputs)
