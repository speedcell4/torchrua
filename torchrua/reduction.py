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
    'tree_reduction_indices',
    'tree_reduce_packed_sequence',
]


def reduce_catted_sequences(sequences: List[Tuple[Tensor, Tensor]],
                            device: Optional[torch.device] = None) -> PackedSequence:
    if device is None:
        device = sequences[0][0].device

    sequence, token_sizes, batch_sizes = zip(*[
        (sequence, token_sizes, token_sizes.size()[0])
        for sequence, token_sizes in sequences
    ])
    sequence = torch.cat(sequence, dim=0).to(device=device)
    token_sizes = torch.cat(token_sizes, dim=0).to(device=device)
    batch_sizes = torch.tensor(batch_sizes, dtype=torch.long, device=device)

    sequence = pack_catted_sequence(sequence=sequence, token_sizes=token_sizes)
    sorting_indices = pack_catted_sequence(sequence=sequence.unsorted_indices, token_sizes=batch_sizes)

    return PackedSequence(
        data=sequence.data,
        batch_sizes=sequence.batch_sizes,
        sorted_indices=invert_permutation(sorting_indices.data),
        unsorted_indices=sorting_indices.data,
    )


class TreeReductionIndices(NamedTuple):
    xs: List[Tensor]
    ys: List[Tensor]
    zs: List[Tensor]
    head: Tensor
    last: Tensor


@torch.no_grad()
def tree_reduction_indices(batch_sizes: Tensor) -> TreeReductionIndices:
    batch_ptr1, token_ptr1, token_sizes1 = token_sizes_to_ptr(token_sizes=batch_sizes)
    acc_batch_sizes1 = accumulate_sizes(sizes=batch_sizes)

    token_sizes2 = token_sizes1 * 2 - 1
    _, token_ptr2, _ = batch_sizes_to_ptr(batch_sizes=token_sizes2)
    acc_token_sizes2 = token_sizes2.cumsum(dim=0)
    last = acc_token_sizes2 - 1
    acc_token_sizes2 = F.pad(acc_token_sizes2, [1, -1])
    offsets = acc_token_sizes2.clone()
    mask = torch.ones_like(token_ptr2, dtype=torch.bool)

    base = 2 ** torch.arange(torch.iinfo(token_sizes1.dtype).bits - 1)
    acc_base = F.pad(base.cumsum(dim=0), [1, -1])
    act_sizes = (token_sizes2[:, None] - acc_base[None, :]).clamp_min(0).min(base)
    act_sizes = act_sizes[:, 1:act_sizes.any(dim=0).long().sum()]

    xs, ys, zs = [], [], []
    for i in range(act_sizes.size()[1] - 1, -1, -1):
        activate_size = act_sizes[:, i]
        token_ptr, batch_ptr, _ = token_sizes_to_ptr(token_sizes=activate_size // 2)
        base_ptr = offsets[batch_ptr] + token_ptr

        x = base_ptr + token_ptr
        z = base_ptr + activate_size[batch_ptr]
        xs.append(x)
        ys.append(x + 1)
        zs.append(z)

        mask[z] = False
        offsets = offsets + activate_size

    head1 = acc_batch_sizes1[token_ptr1] + batch_ptr1
    head2 = acc_token_sizes2[batch_ptr1] + token_ptr2[mask]
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
            (tensor.size()[0] * 2 - last.size()[0], *tensor.size()[1:]),
            dtype=tensor.dtype, device=tensor.device, requires_grad=False,
        )

        data[head] = tensor
        for x, y, z in zip(xs, ys, zs):
            data[z] = fn(data[x], data[y])

        return data[last]

    return _tree_reduce_packed_sequence
