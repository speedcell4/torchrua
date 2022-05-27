import torch
from hypothesis import given
from torch.testing import assert_close

from tests.assertions import assert_packed_sequence_close, assert_grad_close, assert_close
from tests.strategies import devices, sizes, EMBEDDING_DIM, BATCH_SIZE, TOKEN_SIZE
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.reverse import reverse_sequence


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_reverse_catted_sequence(token_sizes, dim, device):
    sequence = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    catted_sequence = cat_sequence(sequence)

    expected = cat_sequence([sequence.flip(dims=[0]) for sequence in sequence])
    actual = reverse_sequence(catted_sequence)

    assert_close(actual.data, expected.data)
    assert_close(actual.token_sizes, expected.token_sizes)
    assert_grad_close(actual.data, expected.data, inputs=sequence)


@given(
    token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
    dim=sizes(EMBEDDING_DIM),
    device=devices(),
)
def test_reverse_packed_sequence(token_sizes, dim, device):
    sequence = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = pack_sequence(sequence)

    expected = pack_sequence([sequence.flip(dims=[0]) for sequence in sequence])
    actual = reverse_sequence(packed_sequence)

    assert_packed_sequence_close(actual, expected)
    assert_grad_close(actual.data, expected.data, inputs=sequence)
