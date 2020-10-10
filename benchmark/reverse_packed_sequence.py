import torch
from tqdm import tqdm

from benchmark.naive_indexing import naive_reverse_packed_sequence
from benchmark.utils import Timer, gen_pack
from torchrua.indexing import reverse_packed_sequence


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
        pack = gen_pack(
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
        pack = gen_pack(
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

# def data1():
#     rua_f = np.array([6.6274, 6.2852, 6.1364, 7.0133, 8.9791]) / 5000
#     rua_b = np.array([11.5218, 12.6473, 18.2495, 34.9357, 70.8818]) / 5000
#
#     naive_f = np.array([7.2876, 8.8388, 9.6415, 9.1631, 13.8260]) / 5000
#     naive_b = np.array([24.9788, 31.8480, 44.1881, 76.4066, 148.5334]) / 5000
#
#     x = [50, 100, 200, 500, 1000]
#
#     return x, rua_f, naive_f, rua_b, naive_b
#
#
# def data2():
#     rua_f = np.array([5.4406, 6.5040, 6.6274, 6.3301, 6.6105, 7.1985]) / 5000
#     rua_b = np.array([5.8177, 12.3177, 11.5218, 15.8664, 28.2674, 47.4997]) / 5000
#
#     naive_f = np.array([3.0818, 5.0671, 7.2876, 12.2371, 41.9305, 78.4146]) / 5000
#     naive_b = np.array([10.5153, 21.0931, 24.9788, 49.2902, 101.2398, 294.3232]) / 5000
#
#     x = [1, 5, 10, 20, 50, 100]
#
#     return x, rua_f, naive_f, rua_b, naive_b
#
#
# from matplotlib import pyplot as plt
# import numpy as np
#
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)  # type:(plt.Figure, (plt.Axes,plt.Axes))
# x, y1, y2, y3, y4 = data1()
# ax1.plot(x, y1, label='rua.forward')
# ax1.plot(x, y3, label='rua.backward')
# ax1.plot(x, y2, label='naive.forward')
# ax1.plot(x, y4, label='naive.backward')
#
# ax1.set_xlim(50, 1000)
# ax1.set_ylim(0, 0.07)
# ax1.set_xlabel('max sentence length')
# ax1.set_ylabel('time (sec)')
# ax1.grid()
# ax1.legend()
#
# x, y1, y2, y3, y4 = data2()
# ax2.plot(x, y1, label='rua.forward')
# ax2.plot(x, y3, label='rua.backward')
# ax2.plot(x, y2, label='naive.forward')
# ax2.plot(x, y4, label='naive.backward')
#
# ax2.set_xlim(1, 100)
# ax2.set_xlabel('batch size')
# ax2.grid()
# ax2.legend()
#
# plt.savefig('..docs/assets/reverse_pack.jpg', bbox_inches='tight', transparent="True")
