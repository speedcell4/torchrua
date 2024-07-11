from torchrua import to_self
from torchrua.layout import C, L, P, R

P.pack = to_self


def to_pack(self: C) -> P:
    z = self.pack_view()
    return z._replace(data=self[z.ptr()])


C.pack = to_pack
L.pack = to_pack
R.pack = to_pack
