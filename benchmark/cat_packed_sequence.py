import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from tqdm import tqdm

from benchmark.utils import Timer, gen_pack
from torchrua.joining import cat_packed_batch_sizes


def rua_forward(rnn: nn.LSTM, pack: PackedSequence, num_chunks: int):
    data = rearrange(pack.data, 'p (d n x) -> (p n) (d x)', d=2 if rnn.bidirectional else 1, n=num_chunks)
    batch_sizes, sorted_indices, unsorted_indices = cat_packed_batch_sizes(pack=pack, num_packs=num_chunks)
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


def cat_pack(num_epoch: int = 1000, batch_size: int = 32, num_chunks: int = 1,
             total_length: int = 120,
             embedding_dim: int = 200, hidden_dim: int = 300, device: int = -1) -> None:
    device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')
    lengths = [
        torch.randint(0, total_length, (batch_size,), device=device) + 1
        for _ in range(num_epoch)
    ]
    rnn = nn.LSTM(
        input_size=embedding_dim // num_chunks, hidden_size=hidden_dim,
        bidirectional=True, bias=True, batch_first=True,
    ).to(device=device)

    rf, rb = Timer(), Timer()
    nf, nb = Timer(), Timer()

    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with rf:
            loss = rua_forward(rnn, pack, num_chunks)
        rnn.zero_grad()
        with rb:
            _ = loss.backward()

    for length in tqdm(lengths):
        pack = gen_pack(
            lengths=length,
            embedding_dim=embedding_dim,
            device=device,
        )
        with nf:
            loss = naive_forward(rnn, pack, num_chunks)
        rnn.zero_grad()
        with nb:
            _ = loss.backward()

    print(f'rua.seconds => {rf.seconds + rb.seconds:.4f} '
          f'({rf.seconds:.4f}, {rb.seconds:.4f})')
    print(f'naive.seconds => {nf.seconds + nb.seconds:.4f} '
          f'({nf.seconds:.4f}, {nb.seconds:.4f})')
