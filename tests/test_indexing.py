import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_packed_close, assert_close
from torchrua.indexing import roll_packed_sequence, reverse_packed_sequence
from torchrua.indexing import select_head, select_last, select_init, select_tail


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_select_head(data, token_sizes, dim, unsort, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    packed_pack_sequence = pack_sequence(sequences, enforce_sorted=False)

    prediction = select_head(sequence=packed_pack_sequence, unsort=unsort)
    if not unsort:
        prediction = prediction[packed_pack_sequence.unsorted_indices]

    target = torch.stack([sequence[0] for sequence in sequences], dim=0)

    assert_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_select_last(data, token_sizes, dim, unsort, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    packed_sequence = pack_sequence(sequences, enforce_sorted=False)

    prediction = select_last(sequence=packed_sequence, unsort=unsort)
    if not unsort:
        prediction = prediction[packed_sequence.unsorted_indices]

    target = torch.stack([sequence[-1] for sequence in sequences], dim=0)

    assert_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_select_init(data, token_sizes, dim, device):
    drop_last_n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))
    sequences = [torch.randn((token_size + 1, dim), device=device) for token_size in token_sizes]
    packed_sequence = pack_sequence(sequences, enforce_sorted=False)

    prediction = select_init(sequence=packed_sequence, drop_last_n=drop_last_n)
    target = pack_sequence([sequence[:-drop_last_n] for sequence in sequences], enforce_sorted=False)

    assert_packed_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_select_tail(data, token_sizes, dim, device):
    drop_first_n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))
    sequences = [torch.randn((token_size + 1, dim), device=device) for token_size in token_sizes]
    packed_sequence = pack_sequence(sequences, enforce_sorted=False)

    prediction = select_tail(sequence=packed_sequence, drop_first_n=drop_first_n)
    target = pack_sequence([sequence[drop_first_n:] for sequence in sequences], enforce_sorted=False)

    assert_packed_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_roll_packed_sequence(data, token_sizes, dim, device):
    offset = data.draw(st.integers(min_value=-max(token_sizes), max_value=+max(token_sizes)))
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    packed_sequence = pack_sequence(sequences, enforce_sorted=False)

    prediction = roll_packed_sequence(sequence=packed_sequence, shifts=offset)
    target = pack_sequence([sequence.roll(offset, dims=[0]) for sequence in sequences], enforce_sorted=False)

    assert_packed_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_reverse_packed_sequence(data, token_sizes, dim, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    packed_sequence = pack_sequence(sequences, enforce_sorted=False)

    prediction = reverse_packed_sequence(sequence=packed_sequence)
    target = pack_sequence([sequence.flip(dims=[0]) for sequence in sequences], enforce_sorted=False)

    assert_packed_close(prediction, target)
