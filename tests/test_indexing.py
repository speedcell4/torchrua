import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from tests.strategies import list_of_sentences
from tests.utils import assert_equal
from torchrua.indexing import batch_indices, token_indices, select_head, select_last, select_init, select_tail, \
    reverse_packed_sequence, roll_packed_sequence


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True)
)
def test_batch_indices(sentences_and_lengths):
    sentences, lengths = sentences_and_lengths
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = batch_indices(pack=pack)

    y = pack_sequence([
        torch.full((length,), fill_value=index, dtype=torch.long, device=x.device)
        for index, length in enumerate(lengths)
    ], enforce_sorted=False).data

    assert_equal(x, y)


@given(
    sentences_and_lengths=list_of_sentences(return_lengths=True)
)
def test_token_indices(sentences_and_lengths):
    sentences, lengths = sentences_and_lengths
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = token_indices(pack=pack)

    y = pack_sequence([
        torch.arange(length, dtype=torch.long, device=x.device)
        for length in lengths
    ], enforce_sorted=False).data

    assert_equal(x, y)


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

    assert_equal(x, y)


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

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(min_value=2),
    drop_last_n=st.integers(1, max_value=11),
)
def test_select_init(sentences, drop_last_n):
    pack = pack_sequence(sentences, enforce_sorted=False)
    drop_last_n = min(drop_last_n, min(len(sent) for sent in sentences) - 1)
    x = select_init(pack=pack, drop_last_n=drop_last_n)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([s[:-drop_last_n] for s in sentences], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(min_value=2),
    drop_first_n=st.integers(1, max_value=11),
)
def test_select_tail(sentences, drop_first_n):
    pack = pack_sequence(sentences, enforce_sorted=False)
    drop_first_n = min(drop_first_n, min(len(sent) for sent in sentences) - 1)
    x = select_tail(pack=pack, drop_first_n=drop_first_n)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([s[drop_first_n:] for s in sentences], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(),
)
def test_reverse_packed_sequence(sentences):
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = reverse_packed_sequence(pack=pack)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([s.flip(dims=[0]) for s in sentences], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert_equal(x, y)


@given(
    sentences=list_of_sentences(),
    offset=st.integers(-5, +5),
)
def test_roll_packed_sequence(sentences, offset):
    pack = pack_sequence(sentences, enforce_sorted=False)
    x = roll_packed_sequence(pack=pack, offset=offset)
    x, _ = pad_packed_sequence(x, batch_first=True)

    y = pack_sequence([s.roll(offset, dims=[0]) for s in sentences], enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    assert_equal(x, y)
