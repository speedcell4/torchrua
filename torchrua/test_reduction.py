import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from torchrua.reduction import tree_reduce_packed_sequence, tree_reduction_indices


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