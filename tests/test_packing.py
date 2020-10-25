import torch
from hypothesis import given
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from tests.strategies import list_of_sentences
from tests.utils import assert_equal
from torchrua.packing import pack_padded_sequence


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True)
)
def test_pack_padded_sequence(sentences_and_lengths):
    sentences, lengths = sentences_and_lengths
    lengths = torch.tensor(lengths, dtype=torch.long, device=sentences[0].device)

    y = pad_sequence(sentences, batch_first=True)
    x = pack_padded_sequence(y, lengths=lengths, batch_first=True, enforce_sorted=False)
    x, _ = pad_packed_sequence(x, batch_first=True)

    assert_equal(x, y)
