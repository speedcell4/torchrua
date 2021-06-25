import torch
from hypothesis import given, strategies as st
from torch import nn
from torch.nn.utils import rnn as tgt

from tests.strategies import token_size_lists, embedding_dims, devices, batch_size_lists, TINY_BATCH_SIZE, \
    TINY_TOKEN_SIZE
from tests.utils import assert_close
from torchrua import cat_sequence, pack_sequence
from torchrua import reduction as rua
from einops import rearrange


@given(
    data=st.data(),
    batch_sizes=batch_size_lists(max_batch_size=TINY_BATCH_SIZE),
    in_dim=embedding_dims(),
    hidden_dim=embedding_dims(),
    device=devices(),
)
def test_reduce_catted_sequences(data, batch_sizes, in_dim, hidden_dim, device):
    sequences = [
        [
            torch.randn((token_size, in_dim), requires_grad=True, device=device)
            for token_size in data.draw(
            token_size_lists(max_token_size=TINY_TOKEN_SIZE, max_batch_size=TINY_BATCH_SIZE))
        ]
        for _ in batch_sizes
    ]

    rnn = nn.LSTM(
        input_size=in_dim,
        hidden_size=hidden_dim,
        bidirectional=True, bias=True,
    ).to(device=device)

    reduction_pack = rua.reduce_catted_sequences([
        cat_sequence(sequence, device=device)
        for sequence in sequences
    ], device=device)
    _, (prediction, _) = rnn(reduction_pack)
    prediction = rearrange(prediction, 'd n x -> n (d x)')

    target = []
    for sequence in sequences:
        _, (t, _) = rnn(pack_sequence(sequence))
        target.append(rearrange(t, 'd n x -> n (d x)'))
    target = pack_sequence(target).data

    assert_close(prediction, target)


@given(
    data=st.data(),
    token_sizes=token_size_lists(),
    dim=embedding_dims(),
    device=devices(),
)
def test_tree_reduce_packed_sequence(data, token_sizes, dim, device):
    sequences = [
        torch.randn((token_size, dim), requires_grad=True, device=device)
        for token_size in token_sizes
    ]

    packed_sequence = tgt.pack_sequence(sequences, enforce_sorted=False)
    reduction_indices = rua.tree_reduction_indices(batch_sizes=packed_sequence.batch_sizes.to(device=device))
    prediction = rua.tree_reduce_packed_sequence(torch.add)(packed_sequence.data, reduction_indices=reduction_indices)

    target = tgt.pad_sequence(sequences, batch_first=False).sum(dim=0)[packed_sequence.sorted_indices]

    assert_close(prediction, target)
