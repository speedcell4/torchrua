import torch
from hypothesis import given, settings, strategies as st
from torch import Tensor
from torchnyan import BATCH_SIZE, FEATURE_DIM, TOKEN_SIZE, assert_grad_close, assert_sequence_close, device, sizes

from torchrua import C, L, Z
from torchrua.reduce import segment_head, segment_last, segment_logsumexp, segment_max, segment_mean, segment_min, \
    segment_prod, segment_sum


def reduce_max(tensor: Tensor) -> Tensor:
    return tensor.max(dim=0).values


def reduce_min(tensor: Tensor) -> Tensor:
    return tensor.min(dim=0).values


def reduce_sum(tensor: Tensor) -> Tensor:
    return tensor.sum(dim=0)


def reduce_mean(tensor: Tensor) -> Tensor:
    return tensor.mean(dim=0)


def reduce_prod(tensor: Tensor) -> Tensor:
    return tensor.prod(dim=0)


def reduce_logsumexp(tensor: Tensor) -> Tensor:
    return tensor.logsumexp(dim=0)


def reduce_head(tensor: Tensor) -> Tensor:
    return tensor[0]


def reduce_last(tensor: Tensor) -> Tensor:
    return tensor[-1]


def raw_segment(sequence, duration, fn):
    expected = []

    for index, duration in enumerate(duration):
        start, end, seq = 0, 0, []
        for size in duration:
            start, end = end, end + size
            seq.append(fn(sequence[index][start:end]))

        seq = torch.stack(seq, dim=0)
        expected.append(seq)

    return expected


@settings(deadline=None)
@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    fns=st.sampled_from([
        (segment_max, reduce_max),
        (segment_min, reduce_min),
        (segment_sum, reduce_sum),
        (segment_mean, reduce_mean),
        (segment_prod, reduce_prod),
        (segment_logsumexp, reduce_logsumexp),
        (segment_head, reduce_head),
        (segment_last, reduce_last),
    ]),
    rua_sequence=st.sampled_from(Z.__args__),
    rua_duration=st.sampled_from(Z.__args__),
)
def test_seg(token_sizes, dim, fns, rua_sequence, rua_duration):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    durations = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), sorted=False, return_counts=True)[1]
        for token_size in token_sizes
    ]

    fn1, fn2 = fns

    actual = rua_sequence.new(inputs).seg(rua_duration.new(durations), fn1).cat()
    expected = C.new(raw_segment(L.new(inputs).data, durations, fn2))

    assert_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
