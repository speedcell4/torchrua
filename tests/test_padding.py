import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from tests.strategies import list_of_sentences
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

    assert torch.equal(x, y), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'


@given(
    sentences=list_of_sentences(),
    unsort=st.booleans(),
    batch_first=st.booleans(),

)
def test_lengths_to_mask(sentences, unsort, batch_first):
    pack = pack_sequence(sentences, enforce_sorted=False)

    x = pack_to_mask(pack, unsort=unsort, batch_first=batch_first, dtype=torch.bool)

    lengths = pack_to_lengths(pack=pack, unsort=unsort, dtype=torch.long)
    y = lengths_to_mask(lengths, filling_mask=True, batch_first=batch_first, dtype=torch.bool)

    assert torch.equal(x, y), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'


@given(
    sentences=list_of_sentences(),
)
def test_lengths_to_batch_sizes(sentences):
    lengths = torch.tensor([s.size(0) for s in sentences], dtype=torch.long, device=sentences[0].device)
    batch_sizes, sorted_indices, unsorted_indices = lengths_to_batch_sizes(lengths=lengths, device=sentences[0].device)
    y = pack_sequence(sentences, enforce_sorted=False)

    assert torch.equal(batch_sizes, y.batch_sizes)
    assert torch.equal(sorted_indices, y.sorted_indices)
    assert torch.equal(unsorted_indices, y.unsorted_indices)
