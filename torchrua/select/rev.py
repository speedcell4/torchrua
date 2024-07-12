import torch

from torchrua.layout import C, L, P, R


def cat_rev(self: C) -> C:
    data, token_sizes = self
    _, _, *sizes = self.size()

    if len(sizes) > 0:
        return self[self.idx().rev()]

    data = torch.flip(data, dims=[0])
    token_sizes = torch.flip(token_sizes, dims=[0])

    data = torch.split(data, token_sizes.detach().cpu().tolist(), dim=0)
    return self._replace(data=torch.cat(data[::-1], dim=0))


C.rev = cat_rev


def left_rev(self: L) -> L:
    return R(data=self.data.flip(dims=[1]), token_sizes=self.token_sizes).left()


L.rev = left_rev


def pack_rev(self: P) -> P:
    return self[self.idx().cat().rev().pack()]


P.rev = pack_rev


def right_rev(self: R) -> R:
    return L(data=self.data.flip(dims=[1]), token_sizes=self.token_sizes).right()


R.rev = right_rev
