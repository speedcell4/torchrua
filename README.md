# TorchRua

![Unit Tests](https://github.com/speedcell4/TorchRua/workflows/Unit%20Tests/badge.svg)

*Rua* derives from the Szechwanese character <ruby>挼<rt>ruá</rt></ruby> which means "pack, rumple, screw up, manipulate". TorchRua provides tons of easy-to-use functions to help you rua variable-length Tensors with `PackedSequence`!

## Requirements

- Python3.7 or higher
- PyTorch 1.6.0

## Performance

* `reverse_packed_sequence`: O(1) forward and O(n) backward with much smaller constant factor than naive implementation.

```python
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from torchrua.indexing import reverse_packed_sequence

x = pack_sequence([
    torch.arange(5) + 1,
    torch.arange(2) + 1,
    torch.arange(3) + 1,
], enforce_sorted=False)
y = reverse_packed_sequence(x)

x, _ = pad_packed_sequence(x, batch_first=True)
y, _ = pad_packed_sequence(y, batch_first=True)

print(x)

# tensor([[1, 2, 3, 4, 5],
#         [1, 2, 0, 0, 0],
#         [1, 2, 3, 0, 0]])

print(y)
# tensor([[5, 4, 3, 2, 1],
#         [2, 1, 0, 0, 0],
#         [3, 2, 1, 0, 0]])
```

<p align="center">
  <img src="assets/reverse_pack.jpg">
</p>