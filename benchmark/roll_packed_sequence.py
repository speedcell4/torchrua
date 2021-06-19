import torch
from tqdm import tqdm

from benchmark.naive import naive_roll_packed_sequence
from benchmark.utils import Timer, gen_pack, report_performance
from torchrua.indexing import roll_packed_sequence


def roll_pack(num_examples: int = 2400, batch_size: int = 32, offset: int = 1,
              total_length: int = 120, embedding_dim: int = 100, device: int = -1):
    device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
    lengths = [
        torch.randint(0, total_length, (batch_size,), device=device) + 1
        for _ in range(num_examples)
    ]

    rua_f = Timer()
    rua_b = Timer()
    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )

        with rua_f:
            y = roll_packed_sequence(pack, shifts=offset).data
        with rua_b:
            y.sum().backward()

    naive_f = Timer()
    naive_b = Timer()
    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with naive_f:
            z = naive_roll_packed_sequence(pack, offset=offset).data
        with naive_b:
            z.sum().backward()

    return report_performance(rua_f, rua_b, naive_f, naive_b)