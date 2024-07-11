from typing import Union

from torchrua.layout import C, L, P, T


def rua(index: Union[T, C, P], sequence: Union[T, C, L, P], *indices: Union[T, C, P]) -> Union[T, C, P]:
    indices = tuple(index.data for index in (index, *indices))
    return index._replace(data=sequence.raw()[indices])


T.rua = rua
C.rua = rua
P.rua = rua
