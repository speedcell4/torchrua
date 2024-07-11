from torchrua import to_self
from torchrua.layout import C, L, P, R

P.pack = to_self


def cat_to_pack(self: C) -> P:
    z = self.pack_view()
    return z._replace(data=self[z.ptr()])


C.pack = cat_to_pack
L.pack = cat_to_pack
R.pack = cat_to_pack
