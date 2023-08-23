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

from torchrua import pad_sequence
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.reduce import segment_max
from torchrua.reduce import segment_mean
from torchrua.reduce import segment_min
from torchrua.reduce import segment_sum


def reduce_mean(x: Tensor) -> Tensor:
    return x.mean(dim=0)


def reduce_sum(x: Tensor) -> Tensor:
    return x.sum(dim=0)


def reduce_max(x: Tensor) -> Tensor:
    return x.max(dim=0).values


def reduce_min(x: Tensor) -> Tensor:
    return x.min(dim=0).values


def raw_segment(tensor, durations, reduce_fn):
    expected = []

    for index, duration in enumerate(durations):
        start, end, seq = 0, 0, []
        for size in duration:
            start, end = end, end + size
            seq.append(reduce_fn(tensor[index][start:end]))

        seq = torch.stack(seq, dim=0)
        expected.append(seq)

    return expected


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(FEATURE_DIM),
    reduce_segment=st.sampled_from([
        (reduce_mean, segment_mean),
        (reduce_sum, segment_sum),
        (reduce_max, segment_max),
        (reduce_min, segment_min),
    ]),
    rua_sequence=st.sampled_from([cat_sequence, pad_sequence, pack_sequence]),
)
def test_segment_padded_sequence(token_sizes, dim, reduce_segment, rua_sequence):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]

    durations = [
        torch.unique(torch.randint(token_size, (token_size,), device=device), sorted=False, return_counts=True)[1]
        for token_size in token_sizes
    ]

    reduce1, reduce2 = reduce_segment

    sequence = pad_sequence(inputs)
    actual = sequence.seg(rua_sequence(durations), reduce2).cat()

    expected = cat_sequence(raw_segment(sequence.data, durations, reduce1))

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
