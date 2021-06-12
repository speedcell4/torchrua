from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from tests.strategies import list_of_homo_lists_of_sentences
from tests.utils import assert_equal
from torchrua import chunk_packed_sequence
from torchrua import stack_packed_sequences


@given(
    lists_of_sentences=list_of_homo_lists_of_sentences(),
    dim=st.sampled_from([0, 1]),
)
def test_chunk_packed_sequence(lists_of_sentences, dim):
    xs = [
        pack_sequence(sentences, enforce_sorted=False)
        for sentences in lists_of_sentences
    ]
    ys = chunk_packed_sequence(
        sequence=stack_packed_sequences(sequences=xs, dim=dim),
        chunks=len(xs), dim=dim,
    )

    for x, y in zip(xs, ys):
        x, _ = pad_packed_sequence(x, batch_first=True)
        y, _ = pad_packed_sequence(y, batch_first=True)

        assert_equal(x, y)
