import torch
from hypothesis import given

from tests.assertions import assert_equal
from tests.strategies import devices, sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.unique import unique_catted_sequence, unique_packed_sequence


@given(
    sequences=sizes(BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM),
    device=devices(),
)
def test_unique_catted_sequence(sequences, device):
    catted_sequence = cat_sequence([torch.tensor(sequence, device=device) for sequence in sequences], device=device)

    unique, inverse, counts = unique_catted_sequence(sequence=catted_sequence, device=device)

    assert_equal(catted_sequence.data, unique[inverse])


@given(
    sequences=sizes(BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM),
    device=devices(),
)
def test_unique_packed_sequence(sequences, device):
    packed_sequence = pack_sequence([torch.tensor(sequence, device=device) for sequence in sequences], device=device)

    unique, inverse, counts = unique_packed_sequence(sequence=packed_sequence, device=device)

    assert_equal(packed_sequence.data, unique[inverse])
