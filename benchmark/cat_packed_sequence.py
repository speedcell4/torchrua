import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from tqdm import tqdm

from benchmark.utils import Timer, gen_pack, report_performance
from torchrua.joining import stack_indices_dim1


def rua_forward(rnn: nn.LSTM, pack: PackedSequence, num_chunks: int):
    data = rearrange(pack.data, 'p (d n x) -> (p n) (d x)', d=2 if rnn.bidirectional else 1, n=num_chunks)
    batch_sizes, sorted_indices, unsorted_indices = stack_indices_dim1(sequence=pack, chunks=num_chunks)
    pack = PackedSequence(
        data=data, batch_sizes=batch_sizes,
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )
    _, (encoding, _) = rnn(pack)
    return encoding.sum()


def naive_forward(rnn: nn.LSTM, pack: PackedSequence, nun_chunks: int):
    data = rearrange(pack.data, 'p (d n x) -> n p (d x)', d=2 if rnn.bidirectional else 1, n=nun_chunks)
    encodings = []
    for index in range(nun_chunks):
        pack = PackedSequence(
            data=data[index], batch_sizes=pack.batch_sizes,
            sorted_indices=pack.sorted_indices,
            unsorted_indices=pack.unsorted_indices,
        )
        _, (encoding, _) = rnn(pack)
        encodings.append(encoding.sum())
    return sum(encodings)


def cat_pack(num_examples: int = 2400, batch_size: int = 32, num_chunks: int = 5,
             total_length: int = 50, embedding_dim: int = 100, hidden_dim: int = 100, device: int = -1):
    device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
    lengths = [
        torch.randint(0, total_length, (batch_size,), device=device) + 1
        for _ in range(num_examples)
    ]
    rnn = nn.LSTM(
        input_size=embedding_dim // num_chunks, hidden_size=hidden_dim,
        bidirectional=True, bias=True, batch_first=True,
    ).to(device=device)

    rua_f, rua_b = Timer(), Timer()
    naive_f, naive_b = Timer(), Timer()

    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with rua_f:
            loss = rua_forward(rnn, pack, num_chunks)
        rnn.zero_grad()
        with rua_b:
            _ = loss.backward()

    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with naive_f:
            loss = naive_forward(rnn, pack, num_chunks)
        rnn.zero_grad()
        with naive_b:
            _ = loss.backward()

    return report_performance(rua_f, rua_b, naive_f, naive_b)
