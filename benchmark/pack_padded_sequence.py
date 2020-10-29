import torch
from torch.nn.utils.rnn import pack_padded_sequence as naive_pack_padded_sequence
from tqdm import tqdm

from benchmark.utils import Timer, gen_pad, report_performance
from torchrua.packing import pack_padded_sequence


def pack_padded(num_examples: int = 2400, batch_size: int = 32,
                total_length: int = 120, embedding_dim: int = 100, device: int = -1):
    device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
    lengths = [
        torch.randint(0, total_length, (batch_size,), device=device) + 1
        for _ in range(num_examples)
    ]

    rua_f = Timer()
    rua_b = Timer()
    for length in tqdm(lengths):
        pad = gen_pad(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )

        with rua_f:
            y = pack_padded_sequence(pad, length, batch_first=True, enforce_sorted=False).data
        with rua_b:
            y.sum().backward()

    naive_f = Timer()
    naive_b = Timer()
    for length in tqdm(lengths):
        pad = gen_pad(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with naive_f:
            z = naive_pack_padded_sequence(pad, length, batch_first=True, enforce_sorted=False).data
        with naive_b:
            z.sum().backward()

    return report_performance(rua_f, rua_b, naive_f, naive_b)
