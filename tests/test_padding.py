import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from tests.strategies import list_of_sentences
from tests.utils import assert_equal
from torchrua.padding import pack_to_lengths, pack_to_mask, lengths_to_mask, lengths_to_batch_sizes


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True),
)
def test_pack_to_lengths(sentences_and_lengths):
    sentences, lengths = sentences_and_lengths

    x = torch.tensor(lengths, dtype=torch.long, device=sentences[0].device)
    y = pack_to_lengths(
        pack=pack_sequence(sentences, enforce_sorted=False),
        unsort=True, dtype=torch.long,
    )

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(),
    unsort=st.booleans(),
    batch_first=st.booleans(),
    filling_mask=st.booleans(),
)
def test_lengths_to_mask(sentences, unsort, batch_first, filling_mask):
    pack = pack_sequence(sentences, enforce_sorted=False)

    x = pack_to_mask(pack, unsort=unsort, filling_mask=filling_mask, batch_first=batch_first, dtype=torch.bool)

    lengths = pack_to_lengths(pack=pack, unsort=unsort, dtype=torch.long)
    y = lengths_to_mask(lengths, filling_mask=filling_mask, batch_first=batch_first, dtype=torch.bool)

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(),
)
def test_lengths_to_batch_sizes(sentences):
    lengths = torch.tensor([s.size(0) for s in sentences], dtype=torch.long, device=sentences[0].device)
    batch_sizes, sorted_indices, unsorted_indices = lengths_to_batch_sizes(lengths=lengths, device=sentences[0].device)
    y = pack_sequence(sentences, enforce_sorted=False)

    assert_equal(batch_sizes, y.batch_sizes)
    assert_equal(sorted_indices, y.sorted_indices)
    assert_equal(unsorted_indices, y.unsorted_indices)
