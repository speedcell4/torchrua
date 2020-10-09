import torch
from torch import Tensor
import sys
import logging
from collections import Counter
from logging import Logger
import math
import random
from pathlib import Path

from typing import Union, Optional, List, Tuple, NamedTuple, Set, Dict, Callable
from typing import Generator, Iterable, KeysView, ValuesView, ItemsView, Any, Type, NewType

import numpy as np
from colorlog import colorlog
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

import torch
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.nn.init import calculate_gain
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch import nn, jit, cuda, initial_seed, autograd, optim, distributions
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from torchglyph.vocab import Vocab, Vectors
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.pipe import Pipe, RawPipe, SeqLengthTensorPipe, PaddedTokLengthPipe
from torchglyph.pipe import IdxTensorPipe, PackedIdxSeqPipe, PackedIdxBlockPipe, PaddedIdxSeqPipe, PaddedIdxBlockPipe
from torchglyph.pipe import TokTensorPipe, PackedTokSeqPipe, PackedTokBlockPipe, PaddedTokSeqPipe, PaddedTokBlockPipe
from torchglyph.pipe import PackedPtrSeqPipe, PackedTokPtrSeqPipe, PackedSeqPtrSeqPipe

from hypothesis import given, strategies as st
from string import ascii_letters, digits

from torch.nn.utils.rnn import PackedSequence, pack_sequence

from torchrua.utils import pack_to_lengths


@torch.no_grad()
def batch_indices(pack: PackedSequence) -> Tensor:
    indices = torch.arange(1, pack.batch_sizes[0].item() + 1)
    if pack.sorted_indices is not None:
        indices = indices[pack.sorted_indices]
    indices = indices[None, :].expand((pack.batch_sizes[0].item(), -1)).tril(0)
    indices = indices[pack.batch_sizes - 1]
    return torch.masked_select(indices, indices != 0) - 1


@torch.no_grad()
def token_indices(pack: PackedSequence) -> Tensor:
    indices = torch.arange(1, pack.batch_sizes.size(0) + 1)
    indices = indices[:, None].expand((-1, pack.batch_sizes[0].item()))

    mask = torch.full((pack.batch_sizes[0].item(),), fill_value=True, dtype=torch.bool)
    if pack.sorted_indices is not None:
        mask = mask[pack.sorted_indices]
    mask = mask[None, :].expand((pack.batch_sizes[0].item(), -1)).tril(0)
    mask = mask[pack.batch_sizes - 1]

    return torch.masked_select(indices, mask) - 1


@torch.no_grad()
def head_indices(pack: PackedSequence, unsort: bool = True) -> Tensor:
    if unsort and pack.unsorted_indices is not None:
        return pack.unsorted_indices
    return torch.arange(0, pack.batch_sizes[0].item(), dtype=torch.long, device=pack.data.device)


def select_head(pack: PackedSequence, unsort: bool = True) -> Tensor:
    return pack.data[head_indices(pack=pack, unsort=unsort)]


@torch.no_grad()
def last_indices(pack: PackedSequence, unsort: bool = True, lengths: Tensor = None) -> Tensor:
    if lengths is None:
        lengths = pack_to_lengths(pack=pack, unsort=False)

    indices = F.pad(pack.batch_sizes.cumsum(dim=0), [2, 0], value=0)[lengths]
    if unsort and pack.unsorted_indices is not None:
        indices = indices[pack.unsorted_indices]
    return indices


def select_last(pack: PackedSequence, unsort: bool = True, lengths: Tensor = None) -> Tensor:
    return pack.data[last_indices(pack=pack, unsort=unsort, lengths=lengths)]


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
    print(select_last(x))
