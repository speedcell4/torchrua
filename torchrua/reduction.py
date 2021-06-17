from typing import Union, Optional, List, Tuple, NamedTuple, Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import invert_permutation

from torchrua.indexing import token_sizes_to_ptr, batch_sizes_to_ptr
from torchrua.packing import pack_catted_sequence
from torchrua.utils import accumulate_sizes

__all__ = [
    'reduce_catted_sequences',
    'tree_reduction_indices', 'tree_reduce_packed_sequence',
]


def reduce_catted_sequences(sequences: List[Tuple[Tensor, Tensor]],
                            device: Optional[torch.device] = None) -> PackedSequence:
    if device is None:
        device = sequences[0][0].device

    data, length1, length2 = zip(*[
        (sequence, lengths, lengths.size()[0])
        for sequence, lengths in sequences
    ])
    data = torch.cat(data, dim=0).to(device=device)
    length1 = torch.cat(length1, dim=0).to(device=device)
    length2 = torch.tensor(length2, dtype=torch.long, device=device)

    data_pack = pack_catted_sequence(sequence=data, token_sizes=length1)
    indices_pack = pack_catted_sequence(sequence=data_pack.unsorted_indices, token_sizes=length2)

    return PackedSequence(
        data=data_pack.data,
        batch_sizes=data_pack.batch_sizes,
        sorted_indices=invert_permutation(indices_pack.data),
        unsorted_indices=indices_pack.data,
    )


class TreeReductionIndices(NamedTuple):
    xs: List[Tensor]
    ys: List[Tensor]
    zs: List[Tensor]
    head: Tensor
    last: Tensor


@torch.no_grad()
def tree_reduction_indices(batch_sizes: Tensor, device: Optional[torch.device]) -> TreeReductionIndices:
    if device is not None:
        device = batch_sizes.device

    _, _, lengths = batch_sizes_to_ptr(
        batch_sizes=batch_sizes,
        sorted_indices=None,
        unsorted_indices=None,
        total_length=None, device=device,
    )
    offsets = torch.zeros_like(lengths)
    acc_batch_sizes1 = accumulate_sizes(batch_sizes)

    lengths2 = lengths * 2 - 1
    batch_ptr2, token_ptr2, batch_sizes2 = token_sizes_to_ptr(
        token_sizes=lengths2,
        sorted_indices=None,
        device=device,
    )
    acc_batch_sizes2 = accumulate_sizes(batch_sizes2)

    mask = torch.ones_like(token_ptr2, dtype=torch.bool)
    offs = torch.zeros_like(mask, dtype=torch.long)
    last = acc_batch_sizes2[lengths2 - 1] + batch_ptr2[:batch_sizes2[0]]

    base = 2 ** torch.arange(torch.iinfo(lengths.dtype).bits - 1)
    acc_base = F.pad(base.cumsum(dim=0), [1, -1])
    clamp_lengths = (lengths2[:, None] - acc_base[None, :]).clamp_min(0).min(base)
    clamp_lengths = clamp_lengths[:, 1:clamp_lengths.any(dim=0).long().sum()].flip(dims=[-1])

    xs, ys, zs = [], [], []
    for i in range(clamp_lengths.size()[1]):
        clamp_lengths_i = clamp_lengths[:, i]
        batch_ptr, token_ptr, _ = token_sizes_to_ptr(clamp_lengths_i // 2, sorted_indices=None, device=device)
        base_ptr = offsets[batch_ptr] + token_ptr

        x = acc_batch_sizes2[base_ptr + token_ptr + 0] + batch_ptr
        y = acc_batch_sizes2[base_ptr + token_ptr + 1] + batch_ptr
        z = acc_batch_sizes2[base_ptr + clamp_lengths_i[batch_ptr]] + batch_ptr
        xs.append(x)
        ys.append(y)
        zs.append(z)

        offs = torch.scatter(offs, dim=0, index=x, src=offsets[batch_ptr])
        offs = torch.scatter(offs, dim=0, index=y, src=offsets[batch_ptr])
        mask = torch.scatter(mask, dim=0, index=z, value=False)
        offsets = offsets + clamp_lengths_i

    batch_ptr2 = batch_ptr2[mask]
    token_ptr2 = token_ptr2[mask]
    token_ptr1 = token_ptr2 - offs[mask] // 2

    head1 = acc_batch_sizes1[token_ptr1] + batch_ptr2
    head2 = acc_batch_sizes2[token_ptr2] + batch_ptr2
    head = head2[invert_permutation(head1)]

    return TreeReductionIndices(xs=xs, ys=ys, zs=zs, head=head, last=last)


def tree_reduce_packed_sequence(fn: Callable[[Tensor, Tensor], Tensor]):
    def _tree_reduce_packed_sequence(tensor: Union[Tensor, PackedSequence],
                                     reduction_indices: TreeReductionIndices) -> Tensor:
        if isinstance(tensor, PackedSequence):
            tensor = tensor.data

        xs, ys, zs, head, last = reduction_indices

        assert tensor.size()[0] == head.size()[0], f'{tensor.size()[0]} != {head.size()[0]}'

        data = torch.empty(
            (last.max().item() + 1, *tensor.size()[1:]),
            dtype=tensor.dtype, device=tensor.device, requires_grad=False,
        )

        data[head] = tensor
        for x, y, z in zip(xs, ys, zs):
            data[z] = fn(data[x], data[y])

        return data[last]

    return _tree_reduce_packed_sequence
