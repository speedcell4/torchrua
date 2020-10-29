import torch
from tqdm import tqdm

from benchmark.naive import naive_roll_packed_sequence
from benchmark.utils import Timer, gen_pack
from torchrua.indexing import roll_packed_sequence


def roll_packed_sequence_fn(num_epoch: int = 5000, batch_size: int = 32, offset: int = 1,
                            total_length: int = 120, embedding_dim: int = 100, device: int = -1) -> None:
    device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
    lengths = [
        torch.randint(0, total_length, (batch_size,), device=device) + 1
        for _ in range(num_epoch)
    ]

    rua_forward_timer = Timer()
    rua_backward_timer = Timer()
    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )

        with rua_forward_timer:
            y = roll_packed_sequence(pack, offset=offset).data
        with rua_backward_timer:
            y.sum().backward()

    naive_forward_timer = Timer()
    naive_backward_timer = Timer()
    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with naive_forward_timer:
            z = naive_roll_packed_sequence(pack, offset=offset).data
        with naive_backward_timer:
            z.sum().backward()

    print(f'rua.seconds => {rua_forward_timer.seconds + rua_backward_timer.seconds:.4f} '
          f'({rua_forward_timer.seconds:.4f}, {rua_backward_timer.seconds:.4f})')
    print(f'naive.seconds => {naive_forward_timer.seconds + naive_backward_timer.seconds:.4f} '
          f'({naive_forward_timer.seconds:.4f}, {naive_backward_timer.seconds:.4f})')
