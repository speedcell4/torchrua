import torch
from hypothesis import given, strategies as st
from torch import Tensor
from torch.testing import assert_close

from tests.assertion import assert_catted_sequence_close, assert_grad_close, assert_packed_sequence_close
from tests.strategy import BATCH_SIZE, device, EMBEDDING_DIM, sizes, TOKEN_SIZE
from torchrua import pad_sequence, segment_sequence
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    reduce=st.sampled_from(['mean', 'sum', 'max', 'min']),
)
def test_segment_catted_sequence(token_sizes, dim, reduce):
    def reduce_fn(x: Tensor) -> Tensor:
        if reduce == 'sum':
            return x.sum(dim=0)
        if reduce == 'mean':
            return x.mean(dim=0)
        if reduce == 'max':
            return x.max(dim=0).values
        if reduce == 'min':
            return x.min(dim=0).values

        raise RuntimeError(f'{reduce} is not supported yet')

    durations = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
        for token_size in token_sizes
    ]

    tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)

    actual = segment_sequence(
        tensor=tensor, reduce=reduce, keep=False,
        sizes=cat_sequence(sequences=durations, device=device),
    )

    expected = []
    for index, duration in enumerate(durations):
        start, end, ans = 0, 0, []
        for size in duration:
            start, end = end, end + size
            ans.append(reduce_fn(tensor[index][start:end]))

        ans = torch.stack(ans, dim=0)
        expected.append(ans)

    expected = cat_sequence(expected)

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=tensor)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    reduce=st.sampled_from(['mean', 'sum', 'max', 'min']),
)
def test_segment_catted_sequence_and_keep(token_sizes, dim, reduce):
    def reduce_fn(x: Tensor) -> Tensor:
        if reduce == 'sum':
            return x.sum(dim=0)
        if reduce == 'mean':
            return x.mean(dim=0)
        if reduce == 'max':
            return x.max(dim=0).values
        if reduce == 'min':
            return x.min(dim=0).values

        raise RuntimeError(f'{reduce} is not supported yet')

    durations = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
        for token_size in token_sizes
    ]

    tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)

    actual = segment_sequence(
        tensor=tensor, reduce=reduce, keep=True,
        sizes=cat_sequence(sequences=durations, device=device),
    )

    expected = []
    for index, duration in enumerate(durations):
        start, end, ans = 0, 0, []
        for size in duration:
            start, end = end, end + size
            ans.append(reduce_fn(tensor[index][start:end]))

        ans = torch.stack(ans, dim=0)
        expected.append(ans)

    expected, _ = pad_sequence(expected, batch_first=True)

    assert_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual, expected=expected, inputs=tensor)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    reduce=st.sampled_from(['mean', 'sum', 'max', 'min']),
)
def test_segment_packed_sequence(token_sizes, dim, reduce):
    def reduce_fn(x: Tensor) -> Tensor:
        if reduce == 'sum':
            return x.sum(dim=0)
        if reduce == 'mean':
            return x.mean(dim=0)
        if reduce == 'max':
            return x.max(dim=0).values
        if reduce == 'min':
            return x.min(dim=0).values

        raise RuntimeError(f'{reduce} is not supported yet')

    durations = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
        for token_size in token_sizes
    ]

    tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)

    actual = segment_sequence(
        tensor=tensor, reduce=reduce, keep=False,
        sizes=pack_sequence(sequences=durations, device=device),
    )

    expected = []
    for index, duration in enumerate(durations):
        start, end, ans = 0, 0, []
        for size in duration:
            start, end = end, end + size
            ans.append(reduce_fn(tensor[index][start:end]))

        ans = torch.stack(ans, dim=0)
        expected.append(ans)

    expected = pack_sequence(expected)

    assert_packed_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=tensor)
