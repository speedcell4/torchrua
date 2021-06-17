from typing import List

import torch
from hypothesis import given, strategies as st
from torch import nn
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from torchrua.catting import cat_sequence
from torchrua.reduction import reduce_catted_sequences
from torchrua.reduction import tree_reduce_packed_sequence, tree_reduction_indices


@given(
    batched_lengths=st.lists(st.lists(st.integers(1, 5), min_size=1, max_size=5), min_size=1, max_size=3),
    dim=st.integers(1, 5),
)
def test_reduce_catted_sequence(batched_lengths: List[List[int]], dim: int):
    data = [
        [torch.randn((length, dim), requires_grad=True) for length in lengths]
        for lengths in batched_lengths
    ]
    rnn = nn.LSTM(input_size=dim, hidden_size=dim, bidirectional=True)

    tgt = []
    for datum in data:
        ys = []
        for x in datum:
            x = pack_sequence([x], enforce_sorted=False)
            _, (y, _) = rnn(x)
            ys.append(y.transpose(0, 1))
        tgt.append(torch.cat(ys, dim=0))
    tgt = pack_sequence(tgt, enforce_sorted=False).data.transpose(0, 1)

    x = reduce_catted_sequences([cat_sequence(datum) for datum in data])
    _, (prd, _) = rnn(x)

    assert torch.allclose(tgt, prd, atol=1e-5, rtol=1e-5)

    for datum in data:
        for x in datum:
            grad_prd, = torch.autograd.grad(
                prd, x, torch.ones_like(prd),
                create_graph=True, allow_unused=False, only_inputs=True,
            )
            grad_tgt, = torch.autograd.grad(
                tgt, x, torch.ones_like(tgt),
                create_graph=True, allow_unused=False, only_inputs=True,
            )
            assert torch.allclose(grad_prd, grad_tgt, atol=1e-5, rtol=1e-5)


@given(st.lists(st.integers(1, 50), min_size=1, max_size=50))
def test_tree_reduce(lengths):
    device = torch.device('cpu')
    pack = pack_sequence([
        torch.randn((length,), dtype=torch.float32, device=device)
        for length in lengths
    ], enforce_sorted=False)
    pad, _ = pad_packed_sequence(pack)
    tgt = pad.sum(dim=0)

    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    sorted_lengths, _ = torch.sort(lengths, dim=0, descending=True)

    prd = tree_reduce_packed_sequence(torch.add)(
        pack.data, tree_reduction_indices(batch_sizes=pack.batch_sizes, device=device))[pack.unsorted_indices]

    assert torch.allclose(tgt, prd, atol=1e-5, rtol=1e-5), f'{tgt} != {prd} '
