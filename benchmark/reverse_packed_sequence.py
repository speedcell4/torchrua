import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

from benchmark.utils import Timer
from torchrua.indexing import reverse_packed_sequence


def naive_reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    data, lengths = pad_packed_sequence(pack, batch_first=True)
    data = [
        data[index, :length].flip(dims=[0])
        for index, length in enumerate(lengths.detach().cpu().tolist())
    ]
    return pack_sequence(data, enforce_sorted=False)


def gen_data(lengths: Tensor, embedding_dim: int, device: torch.device) -> PackedSequence:
    return pack_sequence([
        torch.randn((length, embedding_dim), dtype=torch.float32, device=device, requires_grad=True)
        for length in lengths.detach().cpu().tolist()
    ], enforce_sorted=False)


def reverse_pack_fn(num_epoch: int = 5000, batch_size: int = 32,
                    max_sent_length: int = 120, embedding_dim: int = 100, device: int = -1) -> None:
    device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
    lengths = [
        torch.randint(0, max_sent_length, (batch_size,), device=device) + 1
        for _ in range(num_epoch)
    ]

    rua_forward_timer = Timer()
    rua_backward_timer = Timer()
    for length in tqdm(lengths):
        pack = gen_data(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )

        with rua_forward_timer:
            y = reverse_packed_sequence(pack).data
        with rua_backward_timer:
            y.sum().backward()

    naive_forward_timer = Timer()
    naive_backward_timer = Timer()
    for length in tqdm(lengths):
        pack = gen_data(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with naive_forward_timer:
            z = naive_reverse_packed_sequence(pack).data
        with naive_backward_timer:
            z.sum().backward()

    print(f'rua.seconds => {rua_forward_timer.seconds + rua_backward_timer.seconds:.4f} '
          f'({rua_forward_timer.seconds:.4f}, {rua_backward_timer.seconds:.4f})')
    print(f'naive.seconds => {naive_forward_timer.seconds + naive_backward_timer.seconds:.4f} '
          f'({naive_forward_timer.seconds:.4f}, {naive_backward_timer.seconds:.4f})')
