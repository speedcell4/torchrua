import torch
from hypothesis import given

from tests.assertions import assert_close
from tests.strategies import device, sizes, BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM
from torchrua.catting import cat_sequence
from torchrua.packing import pack_sequence
from torchrua.unique import unique_sequence


@given(
    sequences=sizes(BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM),
)
def test_unique_catted_sequence(sequences):
    catted_sequence = cat_sequence([torch.tensor(sequence, device=device) for sequence in sequences], device=device)

    unique, inverse, counts = unique_sequence(catted_sequence, device=device)

    assert_close(catted_sequence.data, unique[inverse])


@given(
    sequences=sizes(BATCH_SIZE, TOKEN_SIZE, EMBEDDING_DIM),
)
def test_unique_packed_sequence(sequences):
    packed_sequence = pack_sequence([torch.tensor(sequence, device=device) for sequence in sequences], device=device)

    unique, inverse, counts = unique_sequence(packed_sequence, device=device)

    assert_close(packed_sequence.data, unique[inverse])
