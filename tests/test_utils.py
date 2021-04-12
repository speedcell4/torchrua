import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from tests.strategies import list_of_sentences
from tests.utils import assert_equal
from torchrua import packed_sequence_to_mask, lengths_to_mask, packed_sequence_to_lengths, lengths_to_batch_sizes, \
    lengths_to_sorting_indices


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True),
)
def test_packed_sequence_to_lengths(sentences_and_lengths):
    sentences, lengths = sentences_and_lengths

    x = torch.tensor(lengths, dtype=torch.long, device=torch.device('cpu'))
    y = packed_sequence_to_lengths(
        pack=pack_sequence(sentences, enforce_sorted=False),
        unsort=True, dtype=torch.long,
    )

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(),
    unsort=st.booleans(),
    batch_first=st.booleans(),
)
def test_packed_sequence_to_mask(sentences, unsort, batch_first):
    pack = pack_sequence(sentences, enforce_sorted=False)

    x = packed_sequence_to_mask(
        pack=pack, unsort=unsort, batch_first=batch_first, dtype=torch.bool)

    lengths = packed_sequence_to_lengths(pack=pack, unsort=unsort, dtype=torch.long)
    y = lengths_to_mask(lengths, batch_first=batch_first, dtype=torch.bool)

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(),
)
def test_lengths_to_batch_sizes(sentences):
    lengths = torch.tensor([s.size(0) for s in sentences], dtype=torch.long, device=sentences[0].device)
    batch_sizes = lengths_to_batch_sizes(lengths=lengths, device=torch.device('cpu'))
    sorted_indices, unsorted_indices = lengths_to_sorting_indices(lengths=lengths, device=sentences[0].device)
    sorted_sentences = pad_sequence([
        sentences[sorted_index]
        for sorted_index in sorted_indices.detach().cpu().tolist()
    ], batch_first=True)

    data = torch.cat([
        sorted_sentences[:batch_size, index]
        for index, batch_size in enumerate(batch_sizes.detach().cpu().tolist())
    ], dim=0)

    x = PackedSequence(
        data=data, batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pad_sequence(sentences, batch_first=True)

    assert_equal(x, y)
