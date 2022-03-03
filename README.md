# TorchRua

![Unit Tests](https://github.com/speedcell4/TorchRua/workflows/Unit%20Tests/badge.svg)
[![PyPI version](https://badge.fury.io/py/torchrua.svg)](https://badge.fury.io/py/torchrua)
[![Downloads](https://pepy.tech/badge/torchrua)](https://pepy.tech/project/torchrua)

**Rua** derives from the [Szechwanese](https://en.wikipedia.org/wiki/Sichuanese_dialects) character <ruby>挼<rt>
ruá</rt></ruby> which means "pack, rumple, screw up, manipulate". TorchRua provides tons of easy-to-use functions to
help you rua tensors with `PackedSequence` and `CattedSequence`!

## Requirements

- Python 3.8
- PyTorch 1.10.2

## Installation

`python -m pip install torchrua`

## Usage

### Adaptor

* `packed_fn`, `rua_method`, `Packed`

Converting existing function, method, and `nn.Module` to versions that support receiving `PackedSequence` as the first
argument, respectively.

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from torchrua import rua_fn, rua_method, RuaModule
from torchrua.padding import pad_packed_sequence

linear = nn.Linear(7, 13)

pack = pack_sequence([
    torch.randn((5, 7)),
    torch.randn((2, 7)),
    torch.randn((3, 7)),
], enforce_sorted=False)

x, _ = pad_packed_sequence(pack, batch_first=True)
print(x.size())
# torch.Size([3, 5, 7])

print(rua_fn(linear))
# <function packed_fn.<locals>.wrap at 0x10f9524d0>
y, _ = pad_packed_sequence(rua_fn(linear)(pack), batch_first=True)
print(y.size())
# torch.Size([3, 5, 13])

print(rua_method(nn.Linear.forward))
# <function Linear.forward at 0x10f9524d0>
z, _ = pad_packed_sequence(rua_method(nn.Linear.forward)(linear, pack), batch_first=True)
print(z.size())
# torch.Size([3, 5, 13])

print(RuaModule(linear))
# PackedLinear(in_features=7, out_features=13, bias=True)
w, _ = pad_packed_sequence(RuaModule(linear)(pack), batch_first=True)
print(w.size())
# torch.Size([3, 5, 13])
```

* `PackedMeta`
* `PackedSequential`

`PackedMeta` is used to define new `nn.Module`s which support both `PackedSequence` and `torch.Tensor` more
naturally. `PackedSequential` is a packed version `nn.Sequential` which is just defined by using `PackedMeta`.

```python
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_sequence

from torchrua import RuaMeta, RuaSequential
from torchrua.padding import pad_packed_sequence


class MyLinear(nn.Module, metaclass=RuaMeta):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(MyLinear, self).__init__()

        self.weight = nn.Parameter(
            torch.empty((output_dim, input_dim)),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.empty((output_dim,)),
            requires_grad=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input: Tensor):
        return F.linear(input, self.weight, self.bias)


linear = MyLinear(7, 13)
sequential = RuaSequential(
    nn.Linear(7, 13),
    nn.Linear(13, 17),
)

pack = pack_sequence([
    torch.randn((5, 7)),
    torch.randn((2, 7)),
    torch.randn((3, 7)),
], enforce_sorted=False)

x, _ = pad_packed_sequence(pack, batch_first=True)
print(x.size())
# torch.Size([3, 5, 7])

y, _ = pad_packed_sequence(linear(pack), batch_first=True)
print(y.size())
# torch.Size([3, 5, 13])

z, _ = pad_packed_sequence(sequential(pack), batch_first=True)
print(z.size())
# torch.Size([3, 5, 17])
```

### Indexing

* `select_head` (`head_indices`)
* `select_last` (`last_indices`)
* `select_init` (`init_indices`)
* `select_tail` (`tail_indices`)

```python
import torch
from torch.nn.utils.rnn import pack_sequence

from torchrua import head_packed_sequence, last_packed_sequence, init_packed_sequence, tail_packed_sequence
from torchrua.padding import pad_packed_sequence

pack = pack_sequence([
    torch.arange(5) + 1,
    torch.arange(2) + 1,
    torch.arange(3) + 1,
], enforce_sorted=False)
x, _ = pad_packed_sequence(pack, batch_first=True)

print(x)
# tensor([[1, 2, 3, 4, 5],
#         [1, 2, 0, 0, 0],
#         [1, 2, 3, 0, 0]])

print(head_packed_sequence(pack))
# tensor([1, 1, 1])

print(last_packed_sequence(pack))
# tensor([5, 2, 3])

y = init_packed_sequence(pack, n=1)
y, _ = pad_packed_sequence(y, batch_first=True)
print(y)
# tensor([[1, 2, 3, 4],
#         [1, 0, 0, 0],
#         [1, 2, 0, 0]])

z = tail_packed_sequence(pack, n=1)
z, _ = pad_packed_sequence(z, batch_first=True)
print(z)
# tensor([[2, 3, 4, 5],
#         [2, 0, 0, 0],
#         [2, 3, 0, 0]])
```

* `reverse_packed_sequence` (`reversed_indices`)
* `roll_packed_sequence` (`rolled_indices`)

```python
import torch
from torch.nn.utils.rnn import pack_sequence

from torchrua.roll import roll_packed_sequence
from torchrua.reverse import reverse_packed_sequence
from torchrua.padding import pad_packed_sequence

x = pack_sequence([
    torch.arange(5) + 1,
    torch.arange(2) + 1,
    torch.arange(3) + 1,
], enforce_sorted=False)
y = reverse_packed_sequence(x)
z = roll_packed_sequence(x, shifts=+1)
w = roll_packed_sequence(x, shifts=-1)

data, _ = pad_packed_sequence(x, batch_first=True)
print(data)
# tensor([[1, 2, 3, 4, 5],
#         [1, 2, 0, 0, 0],
#         [1, 2, 3, 0, 0]])

data, _ = pad_packed_sequence(y, batch_first=True)
print(data)
# tensor([[5, 4, 3, 2, 1],
#         [2, 1, 0, 0, 0],
#         [3, 2, 1, 0, 0]])

data, _ = pad_packed_sequence(z, batch_first=True)
print(data)
# tensor([[5, 1, 2, 3, 4],
#         [2, 1, 0, 0, 0],
#         [3, 1, 2, 0, 0]])

data, _ = pad_packed_sequence(w, batch_first=True)
print(data)
# tensor([[2, 3, 4, 5, 1],
#         [2, 1, 0, 0, 0],
#         [2, 3, 1, 0, 0]])
```

### Joining & Slicing

* `stack_packed_sequences` (`unstack_packed_sequences`)

If you have several `PackedSequence`s of exactly the same shape, then you can `stack_packed_sequences` them before
feeding them into `nn.LSTM`, joining `PackedSequence`s will significantly accelerate
computing. `unstack_packed_sequence` converts them back to the original `List[PackedSequence]`.

```python
import torch
from torch.nn.utils.rnn import pack_sequence

from torchrua import stack_packed_sequences
from torchrua.padding import pad_packed_sequence

x1 = pack_sequence([
    torch.arange(5) + 1,
    torch.arange(2) + 1,
    torch.arange(3) + 1,
], enforce_sorted=False)

x2 = pack_sequence([
    torch.arange(5) + 11,
    torch.arange(2) + 11,
    torch.arange(3) + 11,
], enforce_sorted=False)

x3 = pack_sequence([
    torch.arange(5) + 21,
    torch.arange(2) + 21,
    torch.arange(3) + 21,
], enforce_sorted=False)

data, _ = pad_packed_sequence(x1, batch_first=True)
print(data)
# tensor([[1, 2, 3, 4, 5],
#         [1, 2, 0, 0, 0],
#         [1, 2, 3, 0, 0]])

data, _ = pad_packed_sequence(x2, batch_first=True)
print(data)
# tensor([[11, 12, 13, 14, 15],
#         [11, 12,  0,  0,  0],
#         [11, 12, 13,  0,  0]])

data, _ = pad_packed_sequence(x3, batch_first=True)
print(data)
# tensor([[21, 22, 23, 24, 25],
#         [21, 22,  0,  0,  0],
#         [21, 22, 23,  0,  0]])

y = stack_packed_sequences([x1, x2, x3], dim=0)
data, _ = pad_packed_sequence(y, batch_first=True)
print(data)
# tensor([[ 1,  2,  3,  4,  5],
#         [11, 12, 13, 14, 15],
#         [21, 22, 23, 24, 25],
#         [ 1,  2,  0,  0,  0],
#         [11, 12,  0,  0,  0],
#         [21, 22,  0,  0,  0],
#         [ 1,  2,  3,  0,  0],
#         [11, 12, 13,  0,  0],
#         [21, 22, 23,  0,  0]])

z = stack_packed_sequences([x1, x2, x3], dim=1)
data, _ = pad_packed_sequence(z, batch_first=True)
print(data)
# tensor([[ 1,  2,  3,  4,  5],
#         [ 1,  2,  0,  0,  0],
#         [ 1,  2,  3,  0,  0],
#         [11, 12, 13, 14, 15],
#         [11, 12,  0,  0,  0],
#         [11, 12, 13,  0,  0],
#         [21, 22, 23, 24, 25],
#         [21, 22,  0,  0,  0],
#         [21, 22, 23,  0,  0]])
```

### Packing

* `pack_sequence`, `pack_padded_sequence`
* `pad_sequence`, `pad_packed_sequence`

These four off-the-shelf alternatives run much faster than the corresponding functions under `torch.nn.utils.rnn`.

```shell script
~ python -m benchmark pack_sequence
PyTorch  (0.0087 sec) = forward (0.0016 sec) + backward (0.0071 sec)
TorchRua (0.0015 sec) = forward (0.0008 sec) + backward (0.0007 sec)

~ python -m benchmark pack_padded_sequence
PyTorch  (0.0055 sec) = forward (0.0010 sec) + backward (0.0045 sec)
TorchRua (0.0011 sec) = forward (0.0006 sec) + backward (0.0005 sec)

~ python -m benchmark pad_sequence
PyTorch  (0.0037 sec) = forward (0.0009 sec) + backward (0.0028 sec)
TorchRua (0.0011 sec) = forward (0.0006 sec) + backward (0.0005 sec)

~ python -m benchmark pad_packed_sequence
PyTorch  (0.0060 sec) = forward (0.0017 sec) + backward (0.0043 sec)
TorchRua (0.0008 sec) = forward (0.0005 sec) + backward (0.0003 sec)
```