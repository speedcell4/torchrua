from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def head_indices(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


def select_head(pack: PackedSequence) -> Tensor:
    raise NotImplementedError


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
