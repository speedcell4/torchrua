from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def naive_reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    data, lengths = pad_packed_sequence(pack, batch_first=True)
    data = [
        data[index, :length].flip(dims=[0])
        for index, length in enumerate(lengths.detach().cpu().tolist())
    ]
    return pack_sequence(data, enforce_sorted=False)


def naive_roll_packed_sequence(pack: PackedSequence, offset: int) -> PackedSequence:
    data, lengths = pad_packed_sequence(pack, batch_first=True)
    data = [
        data[index, :length].roll(offset, dims=[0])
        for index, length in enumerate(lengths.detach().cpu().tolist())
    ]

    return pack_sequence(data, enforce_sorted=False)
