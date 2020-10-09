import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from tests.strategies import list_of_sentences, ATOL, RTOL
from torchrua.indexing import batch_indices, token_indices, select_head, select_last, reverse_packed_sequence


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


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True)
)
def test_token_indices(sentences_and_lengths):
    sentences, lengths = sentences_and_lengths
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = token_indices(pack=pack)

    y = pack_sequence([
        torch.arange(length, dtype=torch.long)
        for length in lengths
    ], enforce_sorted=False).data

    assert torch.equal(x, y), f'{x} != {y}'


@given(
    sentences=list_of_sentences(),
    unsort=st.booleans(),
)
def test_select_head(sentences, unsort):
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = select_head(pack=pack, unsort=unsort)

    y = torch.stack([s[0] for s in sentences], dim=0)
    if not unsort:
        y = y[pack.sorted_indices]

    assert torch.allclose(x, y, rtol=RTOL, atol=ATOL), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'


@given(
    sentences=list_of_sentences(),
    unsort=st.booleans(),
)
def test_select_last(sentences, unsort):
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = select_last(pack=pack, unsort=unsort)

    y = torch.stack([s[-1] for s in sentences], dim=0)
    if not unsort:
        y = y[pack.sorted_indices]

    assert torch.allclose(x, y, rtol=RTOL, atol=ATOL), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'


@given(
    sentences=list_of_sentences(),
)
def test_reverse_packed_sequence(sentences):
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = reverse_packed_sequence(pack=pack)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([s.flip(dims=[0]) for s in sentences], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert torch.allclose(x, y, rtol=RTOL, atol=ATOL), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'
