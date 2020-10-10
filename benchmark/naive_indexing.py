from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_sequence


def naive_reverse_packed_sequence(pack: PackedSequence) -> PackedSequence:
    data, lengths = pad_packed_sequence(pack, batch_first=True)
    data = [
        data[index, :length].flip(dims=[0])
        for index, length in enumerate(lengths.detach().cpu().tolist())
    ]
    return pack_sequence(data, enforce_sorted=False)
