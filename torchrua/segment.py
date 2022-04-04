import torch
from torch import Tensor
from torch.types import Device

from torchrua import CattedSequence, major_sizes_to_ptr


@torch.no_grad()
def segment_catted_indices(sizes: CattedSequence, token_size, device: Device = None):
    if device is None:
        device = sizes.data.device

    sizes, token_sizes = sizes.to(device=device)
    token_ptr, batch_ptr = major_sizes_to_ptr(sizes=token_sizes)

    b, *_ = token_sizes.size()
    t = token_sizes.max().item()

    out = torch.zeros((b, t + 1), dtype=sizes.dtype, device=device)
    out[batch_ptr, token_ptr] = sizes
    out[:, -1] = token_size - out.size(dim=-1)

    return out.view(-1), batch_ptr * (t + 1) + token_ptr


def segment_catted_sequence(tensor: Tensor, reduce: str, sizes: CattedSequence) -> CattedSequence:
    token_sizes, indices = segment_catted_indices(sizes=sizes, token_size=tensor.size()[1], device=tensor.device)

    data = torch.segment_reduce(
        tensor.view((-1, *tensor.size()[2:])),
        reduce=reduce, lengths=token_sizes, unsafe=True,
    )

    return CattedSequence(
        data=data[indices],
        token_sizes=sizes.token_sizes,
    )
