import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def to_mask(pack: PackedSequence, *,
            batch_first: bool = False, padding_value=False,
            dtype: torch.dtype = torch.bool, device: torch.device = None) -> Tensor:
    if dtype is None:
        dtype = pack.data.dtype
    if device is None:
        device = pack.data.device

    mask, _ = pad_packed_sequence(
        PackedSequence(
            data=torch.ones(pack.data.size()[:1], dtype=dtype, device=device),
            batch_sizes=pack.batch_sizes,
            sorted_indices=None if pack.sorted_indices is None else pack.sorted_indices.to(device=device),
            unsorted_indices=None if pack.unsorted_indices is None else pack.unsorted_indices.to(device=device),
        ),
        batch_first=batch_first, padding_value=padding_value,
    )
    return mask


if __name__ == '__main__':
    x = pack_sequence([
        torch.randn((5, 2)),
        torch.randn((2, 2)),
        torch.randn((3, 2)),
    ], enforce_sorted=False)
    print(to_mask(x, batch_first=True))
