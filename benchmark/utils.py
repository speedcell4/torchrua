from datetime import datetime

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_sequence


class Timer(object):
    def __init__(self):
        super(Timer, self).__init__()
        self.seconds = 0

    def __enter__(self):
        self.start_tm = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seconds += (datetime.now() - self.start_tm).total_seconds()
        del self.start_tm


def report_performance(rua_f, rua_b, naive_f, naive_b):
    rua_f = rua_f.seconds
    rua_b = rua_b.seconds
    print(f'torchrua (sec) => {rua_f + rua_b:.4f} = {rua_f:.4f} (forward) + {rua_b:.4f} (backward)')

    naive_f = naive_f.seconds
    naive_b = naive_b.seconds
    print(f'naive (sec) => {naive_f + naive_b:.4f} = {naive_f:.4f} (forward) + {naive_b:.4f} (backward)')

    return (rua_f, rua_b), (naive_f, naive_b)


def gen_pad(lengths: Tensor, embedding_dim: int, device: torch.device) -> PackedSequence:
    return pad_sequence([
        torch.randn((length, embedding_dim), dtype=torch.float32, device=device, requires_grad=True)
        for length in lengths.detach().cpu().tolist()
    ], batch_first=True)


def gen_pack(lengths: Tensor, embedding_dim: int, device: torch.device) -> PackedSequence:
    return pack_sequence([
        torch.randn((length, embedding_dim), dtype=torch.float32, device=device, requires_grad=True)
        for length in lengths.detach().cpu().tolist()
    ], enforce_sorted=False)
