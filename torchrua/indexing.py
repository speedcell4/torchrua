import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True) -> Tensor:
    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices
    return torch.arange(0, pack.batch_sizes[0].item(), dtype=torch.long, device=pack.data.device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


def last_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_last(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def init_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_init(pack: PackedSequence) -> PackedSequence:
    raise NotImplementedError


def tail_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_tail(pack: PackedSequence) -> PackedSequence:
    raise NotImplementedError


def reverse_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def flip_packed_sequence(pack: PackedSequence) -> PackedSequence:
    raise NotImplementedError


if __name__ == '__main__':
    x = pack_sequence([
        torch.arange(5),
        torch.arange(2) + 5,
        torch.arange(3) + 5 + 2,
    ], enforce_sorted=False)
    data, _ = pad_packed_sequence(x, batch_first=True)
    print(data)
    print(head_indices(x, unsort=True))
    print(select_head(x, unsort=True))
    print(head_indices(x, unsort=False))
    print(select_head(x, unsort=False))
