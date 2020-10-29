# TorchRua

![Unit Tests](https://github.com/speedcell4/TorchRua/workflows/Unit%20Tests/badge.svg)
[![PyPI version](https://badge.fury.io/py/torchrua.svg)](https://badge.fury.io/py/torchrua)
[![Downloads](https://pepy.tech/badge/torchrua)](https://pepy.tech/project/torchrua)

*Rua* derives from the Szechwanese character <ruby>挼<rt>ruá</rt></ruby> which means "pack, rumple, screw up, manipulate". TorchRua provides tons of easy-to-use functions to help you rua variable-length Tensors with `PackedSequence`s!

## Requirements

- Python 3.7 or higher
- PyTorch 1.6.0 or higher

## Install

`python -m pip install torchrua --upgrade`

## Usage

### Adaptor

* `packed_fn`, `packed_method`, `Packed`

Converting existing function, method, and `nn.Module` to versions that support receiving `PackedSequence` as the first argument, respectively.

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from torchrua import packed_fn, packed_method, Packed, pad_packed_sequence

linear = nn.Linear(7, 13)

pack = pack_sequence([
    torch.randn((5, 7)),
    torch.randn((2, 7)),
    torch.randn((3, 7)),
], enforce_sorted=False)

x, _ = pad_packed_sequence(pack, batch_first=True)
print(x.size())
# torch.Size([3, 5, 7])

print(packed_fn(linear))
# <function packed_fn.<locals>.wrap at 0x10f9524d0>
y, _ = pad_packed_sequence(packed_fn(linear)(pack), batch_first=True)
print(y.size())
# torch.Size([3, 5, 13])

print(packed_method(nn.Linear.forward))
# <function Linear.forward at 0x10f9524d0>
z, _ = pad_packed_sequence(packed_method(nn.Linear.forward)(linear, pack), batch_first=True)
print(z.size())
# torch.Size([3, 5, 13])

print(Packed(linear))
# PackedLinear(in_features=7, out_features=13, bias=True)
w, _ = pad_packed_sequence(Packed(linear)(pack), batch_first=True)
print(w.size())
# torch.Size([3, 5, 13])
```

* `PackedMeta`
* `PackedSequential`

`PackedMeta` is used to define new `nn.Module`s which support both `PackedSequence` and `torch.Tensor` more naturally. `PackedSequential` is a packed version `nn.Sequential` which is just defined by using `PackedMeta`.

```python
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_sequence

from torchrua import PackedMeta, pad_packed_sequence, PackedSequential


class MyLinear(nn.Module, metaclass=PackedMeta):
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
sequential = PackedSequential(
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

from torchrua import select_head, select_last, select_init, select_tail, pad_packed_sequence

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

print(select_head(pack))
# tensor([1, 1, 1])

print(select_last(pack))
# tensor([5, 2, 3])

y = select_init(pack, drop_last_n=1)
y, _ = pad_packed_sequence(y, batch_first=True)
print(y)
# tensor([[1, 2, 3, 4],
#         [1, 0, 0, 0],
#         [1, 2, 0, 0]])

z = select_tail(pack, drop_first_n=1)
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

from torchrua import reverse_packed_sequence, roll_packed_sequence, pad_packed_sequence

x = pack_sequence([
    torch.arange(5) + 1,
    torch.arange(2) + 1,
    torch.arange(3) + 1,
], enforce_sorted=False)
y = reverse_packed_sequence(x)
z = roll_packed_sequence(x, offset=+1)
w = roll_packed_sequence(x, offset=-1)

x, _ = pad_packed_sequence(x, batch_first=True)
print(x)
# tensor([[1, 2, 3, 4, 5],
#         [1, 2, 0, 0, 0],
#         [1, 2, 3, 0, 0]])

y, _ = pad_packed_sequence(y, batch_first=True)
print(y)
# tensor([[5, 4, 3, 2, 1],
#         [2, 1, 0, 0, 0],
#         [3, 2, 1, 0, 0]])

z, _ = pad_packed_sequence(z, batch_first=True)
print(z)
# tensor([[5, 1, 2, 3, 4],
#         [2, 1, 0, 0, 0],
#         [3, 1, 2, 0, 0]])

w, _ = pad_packed_sequence(w, batch_first=True)
print(w)
# tensor([[2, 3, 4, 5, 1],
#         [2, 1, 0, 0, 0],
#         [2, 3, 1, 0, 0]])
```

```shell script
~ python -m benchmark reverse_pack --device 0
torchrua (sec) => 13.9069 = 3.2287 (forward) + 10.6782 (backward)
naive (sec) => 43.9063 = 8.6442 (forward) + 35.2621 (backward)

~ python -m benchmark roll_pack --device 0
torchrua (sec) => 14.5345 = 3.3659 (forward) + 11.1686 (backward)
naive (sec) => 46.0561 = 8.9097 (forward) + 37.1464 (backward)
```

### Joining & Slicing

* `cat_packed_sequences` (`uncat_packed_sequences`)
* `stack_packed_sequences` (`unstack_packed_sequences`)

If you have several `PackedSequence`s of exactly the same shape, then you can `cat_packed_sequences` or `stack_packed_sequences` them before feeding them into `nn.LSTM`, joining `PackedSequence`s will significantly accelerate computing. `uncat_packed_sequence` and `unstack_packed_sequence` converts them back to the original `List[PackedSequence]`.

```python
import torch
from torch.nn.utils.rnn import pack_sequence

from torchrua import cat_packed_sequences, stack_packed_sequences, pad_packed_sequence

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

y = cat_packed_sequences([x1, x2, x3])

z = stack_packed_sequences([x1, x2, x3])

x1, _ = pad_packed_sequence(x1, batch_first=True)
print(x1)
# tensor([[1, 2, 3, 4, 5],
#         [1, 2, 0, 0, 0],
#         [1, 2, 3, 0, 0]])

x2, _ = pad_packed_sequence(x2, batch_first=True)
print(x2)
# tensor([[11, 12, 13, 14, 15],
#         [11, 12,  0,  0,  0],
#         [11, 12, 13,  0,  0]])

x3, _ = pad_packed_sequence(x3, batch_first=True)
print(x3)
# tensor([[21, 22, 23, 24, 25],
#         [21, 22,  0,  0,  0],
#         [21, 22, 23,  0,  0]])

y, _ = pad_packed_sequence(y, batch_first=True)
print(y)
# tensor([[ 1,  2,  3,  4,  5],
#         [ 1,  2,  0,  0,  0],
#         [ 1,  2,  3,  0,  0],
#         [11, 12, 13, 14, 15],
#         [11, 12,  0,  0,  0],
#         [11, 12, 13,  0,  0],
#         [21, 22, 23, 24, 25],
#         [21, 22,  0,  0,  0],
#         [21, 22, 23,  0,  0]])

z, _ = pad_packed_sequence(z, batch_first=True)
print(z)
# tensor([[ 1,  2,  3,  4,  5],
#         [11, 12, 13, 14, 15],
#         [21, 22, 23, 24, 25],
#         [ 1,  2,  0,  0,  0],
#         [11, 12,  0,  0,  0],
#         [21, 22,  0,  0,  0],
#         [ 1,  2,  3,  0,  0],
#         [11, 12, 13,  0,  0],
#         [21, 22, 23,  0,  0]])
```

```shell script
~ python -m benchmark cat_pack --device 0
torchrua (sec) => 23.5499 = 8.6669 (forward) + 14.8830 (backward)
naive (sec) => 72.9520 = 30.8352 (forward) + 42.1168 (backward)
```

### Packing

* `pack_padded_sequence`
* `pad_packed_sequence`

These two functions are the same as `torch.nn.utils.rnn.pack_padded_sequence` and `torch.nn.utils.rnn.pad_packed_sequence`, with sightly lower speed on forward pass but much higher speed on backward pass.


```shell script
~ python -m benchmark pack_padded --device 0
torchrua (sec) => 9.2755 = 2.4933 (forward) + 6.7822 (backward)
naive (sec) => 11.9262 = 1.9409 (forward) + 9.9852 (backward)

~ python -m benchmark pad_packed --device 0
torchrua (sec) => 14.1087 = 2.5054 (forward) + 11.6034 (backward)
naive (sec) => 21.5252 = 2.5133 (forward) + 19.0119 (backward)
```