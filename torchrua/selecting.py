import torch
from torch import Tensor
from torchrua.catting import cat_sequence

from torchrua import CattedSequence
from torchrua.utils import accumulate_sizes


@torch.no_grad()
def head_catted_indices(sequence: CattedSequence) -> Tensor:
    return accumulate_sizes(sizes=sequence.token_sizes)


@torch.no_grad()
def last_catted_indices(sequence: CattedSequence) -> Tensor:
    raise NotImplementedError


@torch.no_grad()
def init_catted_indices(sequence: CattedSequence, drop_n: int = 1) -> Tensor:
    raise NotImplementedError


@torch.no_grad()
def tail_catted_indices(sequence: CattedSequence, drop_n: int = 1) -> Tensor:
    raise NotImplementedError


def head_catted_sequence(sequence: CattedSequence) -> Tensor:
    indices = head_catted_indices(sequence=sequence)
    return sequence.data[indices]


def last_catted_sequence(sequence: CattedSequence) -> Tensor:
    raise NotImplementedError


def init_catted_sequence(sequence: CattedSequence, drop_n: int = 1) -> CattedSequence:
    raise NotImplementedError


def tail_catted_sequence(sequence: CattedSequence, drop_n: int = 1) -> CattedSequence:
    raise NotImplementedError


if __name__ == '__main__':
    data = cat_sequence([
        torch.arange(5),
        torch.arange(2),
        torch.arange(3),
    ])
    print(data)
    print(head_catted_sequence(data))
