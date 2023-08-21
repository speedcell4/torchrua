from typing import Any
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Type
from typing import Union

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


class CattedSequence(NamedTuple):
    data: Tensor
    token_sizes: Tensor

    def to(self, *args, **kwargs) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.to(*args, **kwargs),
            token_sizes=self.token_sizes.to(*args, **kwargs),
        )

    def cpu(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.cpu(),
            token_sizes=self.token_sizes.cpu(),
        )

    def cuda(self) -> 'CattedSequence':
        return CattedSequence(
            data=self.data.cuda(),
            token_sizes=self.token_sizes.cuda(),
        )


T = Tensor
D = Tuple[Tensor, Tensor]
C = CattedSequence
P = PackedSequence

Sequence = Union[T, D, C, P]

Ts = List[T]
Ds = List[D]
Cs = List[C]
Ps = List[P]

Sequences = Union[Ts, Ds, Cs, Ps]


def is_instance(obj: Any, ty: Type) -> bool:
    __origin__ = getattr(ty, '__origin__', None)
    __args__ = getattr(ty, '__args__', [])

    if __origin__ is list:
        if not isinstance(obj, list):
            return False

        return all(is_instance(o, __args__[0]) for o in obj)

    if __origin__ is tuple:
        if not isinstance(obj, tuple):
            return False

        if len(__args__) == 2 and __args__[1] is ...:
            return all(is_instance(o, __args__[0]) for o in obj)

        if len(__args__) == len(obj):
            return all(is_instance(o, t) for o, t in zip(obj, __args__))

        return False

    if __origin__ is Union:
        return any(is_instance(obj, t) for t in __args__)

    return isinstance(obj, ty)
