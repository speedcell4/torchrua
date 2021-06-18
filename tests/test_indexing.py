import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence

from tests.strategies import token_size_lists, embedding_dims, devices
from tests.utils import assert_packed_close, assert_close
from torchrua import select_head, select_last, select_init, select_tail


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_select_head(data, token_sizes, dim, unsort, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    pack = pack_sequence(sequences, enforce_sorted=False)

    prd = select_head(pack=pack, unsort=unsort)

    tgt = torch.stack([sequence[0] for sequence in sequences], dim=0)
    if not unsort:
        tgt = tgt[pack.sorted_indices]

    assert_close(prd, tgt)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    unsort=st.booleans(),
    device=devices(),
)
def test_select_last(data, token_sizes, dim, unsort, device):
    sequences = [torch.randn((token_size, dim), device=device) for token_size in token_sizes]
    pack = pack_sequence(sequences, enforce_sorted=False)

    prd = select_last(pack=pack, unsort=unsort)

    tgt = torch.stack([sequence[-1] for sequence in sequences], dim=0)
    if not unsort:
        tgt = tgt[pack.sorted_indices]

    assert_close(prd, tgt)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_select_init(data, token_sizes, dim, device):
    drop_last_n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))
    sequences = [torch.randn((token_size + 1, dim), device=device) for token_size in token_sizes]
    pack = pack_sequence(sequences, enforce_sorted=False)

    prd = select_init(pack=pack, drop_last_n=drop_last_n)
    tgt = pack_sequence([sequence[:-drop_last_n] for sequence in sequences], enforce_sorted=False)

    assert_packed_close(prd, tgt)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_select_tail(data, token_sizes, dim, device):
    drop_first_n = data.draw(st.integers(min_value=1, max_value=min(token_sizes)))
    sequences = [torch.randn((token_size + 1, dim), device=device) for token_size in token_sizes]
    pack = pack_sequence(sequences, enforce_sorted=False)

    prd = select_tail(pack=pack, drop_first_n=drop_first_n)
    tgt = pack_sequence([sequence[drop_first_n:] for sequence in sequences], enforce_sorted=False)

    assert_packed_close(prd, tgt)
