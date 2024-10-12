import torch

from torchrua.layout import C, L, P, R, Z


def cat_seg(self: C, duration: Z, fn) -> C:
    duration = duration.cat()
    data = fn(self.data, duration.data)

    return duration._replace(data=data)


C.seg = cat_seg


def left_seg(self: L, duration: Z, fn) -> L:
    duration = duration.left(0)

    b, t, *sizes = self.size()
    token_sizes = torch.cat([duration.data, t - self.token_sizes[:, None]], dim=-1).view(-1)

    data = fn(self.data.flatten(start_dim=0, end_dim=1), token_sizes)
    data = data.view((b, -1, *sizes))

    return L(data=data[:, :-1], token_sizes=duration.token_sizes)


L.seg = left_seg


def pack_seg(self: P, duration: Z, fn) -> P:
    return self.cat().seg(duration, fn).pack()


P.seg = pack_seg


def right_seg(self: R, duration: Z, fn) -> R:
    duration = duration.right(0)

    b, t, *sizes = self.size()
    token_sizes = torch.cat([t - self.token_sizes[:, None], duration.data], dim=-1).view(-1)

    data = fn(self.data.flatten(start_dim=0, end_dim=1), token_sizes)
    data = data.view((b, -1, *sizes))

    return R(data=data[:, +1:], token_sizes=duration.token_sizes)


R.seg = right_seg
