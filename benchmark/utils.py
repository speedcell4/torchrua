from datetime import datetime

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence


class Timer(object):
    def __init__(self):
        super(Timer, self).__init__()
        self.seconds = 0

    def __enter__(self):
        self.start_tm = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seconds += (datetime.now() - self.start_tm).total_seconds()
        del self.start_tm


def gen_pack(lengths: Tensor, embedding_dim: int, device: torch.device) -> PackedSequence:
    return pack_sequence([
        torch.randn((length, embedding_dim), dtype=torch.float32, device=device, requires_grad=True)
        for length in lengths.detach().cpu().tolist()
    ], enforce_sorted=False)