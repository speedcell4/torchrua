import torch
from hypothesis import given

from tests.strategies import draw_token_sizes, draw_device, draw_embedding_dim
from tests.utils import assert_equal
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.unique import unique_catted_sequence, unique_packed_sequence


@given(
    vocab_size=draw_embedding_dim(),
    token_sizes=draw_token_sizes(),
    device=draw_device(),
)
def test_unique_catted_sequence(vocab_size, token_sizes, device):
    sequence = cat_sequence([
        torch.randint(vocab_size, (token_size,), dtype=torch.long, device=device)
        for token_size in token_sizes
    ])

    unique, inverse, counts = unique_catted_sequence(sequence=sequence, device=device)

    assert_equal(sequence.data, unique[inverse])


@given(
    vocab_size=draw_embedding_dim(),
    token_sizes=draw_token_sizes(),
    device=draw_device(),
)
def test_unique_packed_sequence(vocab_size, token_sizes, device):
    sequence = pack_sequence([
        torch.randint(vocab_size, (token_size,), dtype=torch.long, device=device)
        for token_size in token_sizes
    ])

    unique, inverse, counts = unique_packed_sequence(sequence=sequence, device=device)

    assert_equal(sequence.data, unique[inverse])
