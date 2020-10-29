import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pad_packed_sequence as pad_packed_sequence_naive
from torch.nn.utils.rnn import pad_sequence, pack_sequence

from tests.strategies import list_of_sentences
from tests.utils import assert_equal
from torchrua import pad_packed_sequence, pack_padded_sequence


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True),
    batch_first=st.booleans(),
)
def test_pack_padded_sequence(sentences_and_lengths, batch_first):
    sentences, lengths = sentences_and_lengths
    lengths = torch.tensor(lengths, dtype=torch.long, device=sentences[0].device)

    y = pad_sequence(sentences, batch_first=batch_first)
    x = pack_padded_sequence(y, lengths=lengths, batch_first=batch_first, enforce_sorted=False)
    x, _ = pad_packed_sequence_naive(x, batch_first=batch_first)

    assert_equal(x, y)


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True),
    batch_first=st.booleans(),
)
def test_pack_padded_sequence(sentences_and_lengths, batch_first):
    sentences, lengths = sentences_and_lengths

    x_lengths = torch.tensor(lengths, dtype=torch.long, device=torch.device('cpu'))

    pack = pack_sequence(sentences, enforce_sorted=False)
    y_data, y_lengths = pad_packed_sequence(pack, batch_first=batch_first)

    x_data = pad_sequence(sentences, batch_first=batch_first)

    assert_equal(x_data, y_data)
    assert_equal(x_lengths, y_lengths)
