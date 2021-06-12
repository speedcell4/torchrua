from hypothesis import given
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from tests.strategies import list_of_homo_lists_of_sentences
from tests.utils import assert_equal
from torchrua import stack_packed_sequences


@given(
    lists_of_sentences=list_of_homo_lists_of_sentences()
)
def test_stack_packed_sequences_dim0(lists_of_sentences):
    xs = [
        pack_sequence(sentences, enforce_sorted=False)
        for sentences in lists_of_sentences
    ]
    x = stack_packed_sequences(sequences=xs, dim=0)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([
        s for sentences in list(zip(*lists_of_sentences)) for s in sentences
    ], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert_equal(x, y)


@given(
    lists_of_sentences=list_of_homo_lists_of_sentences()
)
def test_stack_packed_sequences_dim1(lists_of_sentences):
    xs = [
        pack_sequence(sentences, enforce_sorted=False)
        for sentences in lists_of_sentences
    ]
    x = stack_packed_sequences(sequences=xs, dim=1)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([
        s for sentences in lists_of_sentences for s in sentences
    ], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert_equal(x, y)
