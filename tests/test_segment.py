import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor
from torchnyan import BATCH_SIZE
from torchnyan import FEATURE_DIM
from torchnyan import TOKEN_SIZE
from torchnyan import assert_catted_sequence_close
from torchnyan import assert_grad_close
from torchnyan import device
from torchnyan import sizes

from torchrua import cat_sequence
from torchrua import pack_sequence
from torchrua import pad_sequence
from torchrua import segment_logsumexp
from torchrua import segment_max
from torchrua import segment_mean
from torchrua import segment_min
from torchrua import segment_prod
from torchrua import segment_sum


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


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    fns=st.sampled_from([
        (reduce_max, segment_max),
        (reduce_min, segment_min),
        (reduce_sum, segment_sum),
        (reduce_mean, segment_mean),
        (reduce_prod, segment_prod),
        (reduce_logsumexp, segment_logsumexp),
    ]),
    rua_sequence=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
    rua_duration=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
)
def test_segment_sequence(token_sizes, dim, fns, rua_sequence, rua_duration):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    durations = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), sorted=False, return_counts=True)[1]
        for token_size in token_sizes
    ]

    fn1, fn2 = fns

    sequence = rua_sequence(inputs)
    actual = sequence.seg(rua_duration(durations), fn2).cat()

    expected = cat_sequence(raw_segment(pad_sequence(inputs).data, durations, fn1))

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
