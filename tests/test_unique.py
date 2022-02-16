import torch
from hypothesis import given

from tests.strategies import token_size_lists, devices, embedding_dims
from tests.utils import assert_equal
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.unique import unique_catted_sequence, unique_packed_sequence


@given(
    vocab_size=embedding_dims(),
    token_sizes=token_size_lists(),
    device=devices(),
)
def test_unique_catted_sequence(vocab_size, token_sizes, device):
    sequence = cat_sequence([
        torch.randint(vocab_size, (token_size,), dtype=torch.long, device=device)
        for token_size in token_sizes
    ])

    unique, inverse, counts = unique_catted_sequence(sequence=sequence, device=device)

    assert_equal(sequence.data, unique[inverse])


@given(
    vocab_size=embedding_dims(),
    token_sizes=token_size_lists(),
    device=devices(),
)
def test_unique_packed_sequence(vocab_size, token_sizes, device):
    sequence = pack_sequence([
        torch.randint(vocab_size, (token_size,), dtype=torch.long, device=device)
        for token_size in token_sizes
    ])

    unique, inverse, counts = unique_packed_sequence(sequence=sequence, device=device)

    assert_equal(sequence.data, unique[inverse])
