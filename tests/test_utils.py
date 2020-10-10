import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from tests.strategies import list_of_sentences
from torchrua.utils import pack_to_lengths, pack_to_mask, lengths_to_mask


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True),
)
def test_pack_to_lengths(sentences_and_lengths):
    sentences, x = sentences_and_lengths

    x = torch.tensor(x, dtype=torch.long, device=sentences[0].device)
    y = pack_to_lengths(
        pack=pack_sequence(sentences, enforce_sorted=False),
        unsort=True, dtype=torch.long,
    )

    assert torch.equal(x, y), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'


@given(
    sentences_and_lengths=list_of_sentences(),
    unsort=st.booleans(),
    batch_first=st.booleans(),

)
def test_lengths_to_mask(sentences_and_lengths, unsort, batch_first):
    sentences = sentences_and_lengths
    pack = pack_sequence(sentences, enforce_sorted=False)

    x = pack_to_mask(pack, unsort=unsort, batch_first=batch_first, dtype=torch.bool)

    lengths = pack_to_lengths(pack=pack, unsort=unsort, dtype=torch.long)
    y = lengths_to_mask(lengths, filling_mask=True, batch_first=batch_first, dtype=torch.bool)

    assert torch.equal(x, y), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'
