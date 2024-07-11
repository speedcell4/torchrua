from typing import Union

from torchrua.layout import C, L, P, R
from torchrua.utils import to_self

C.cat = to_self


def to_cat(self: Union[L, P, R]) -> C:
    z = self.cat_view()
    return z._replace(data=self[z.ptr()])


L.cat = to_cat
R.cat = to_cat
P.cat = to_cat
