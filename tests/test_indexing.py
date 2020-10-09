import torch
from hypothesis import given
from torch.nn.utils.rnn import pack_sequence

from tests.strategies import list_of_sentences
from torchrua.indexing import batch_indices


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True)
)
def test_batch_indices(sentences_and_lengths):
    sentences, lengths = sentences_and_lengths
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = batch_indices(pack=pack)

    y = pack_sequence([
        torch.full((length,), fill_value=index, dtype=torch.long)
        for index, length in enumerate(lengths)
    ], enforce_sorted=False).data

    assert torch.equal(x, y), f'{x} != {y}'
