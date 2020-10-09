import torch
from hypothesis import given
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from tests.strategies import list_of_homo_lists_of_sentences, RTOL, ATOL
from torchrua.joining import cat_packed_sequences, stack_packed_sequences


@given(
    lists_of_sentences=list_of_homo_lists_of_sentences()
)
def test_cat_packed_sequences(lists_of_sentences):
    xs = [
        pack_sequence(sentences, enforce_sorted=False)
        for sentences in lists_of_sentences
    ]
    x = cat_packed_sequences(packs=xs)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([
        s for sentences in lists_of_sentences for s in sentences
    ], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert torch.allclose(x, y, rtol=RTOL, atol=ATOL), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'


@given(
    lists_of_sentences=list_of_homo_lists_of_sentences()
)
def test_stack_packed_sequences(lists_of_sentences):
    xs = [
        pack_sequence(sentences, enforce_sorted=False)
        for sentences in lists_of_sentences
    ]
    x = stack_packed_sequences(packs=xs)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([
        s for sentences in list(zip(*lists_of_sentences)) for s in sentences
    ], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert torch.allclose(x, y, rtol=RTOL, atol=ATOL), f'{x.contiguous().view(-1)} != {y.contiguous().view(-1)}'
