# from typing import Tuple
#
# from torchrua.layout import C, L, P, R, T, is_type
#
#
# def __getitem__c(sequence: C, ptr: Tuple[T, T]) -> C:
#     if is_type(ptr, Tuple[T, T]):
#         batch_ptr, token_ptr = ptr
#         return sequence._replace(data=sequence.data[sequence.offsets()[batch_ptr] + token_ptr])
#
#     return super(C, sequence).__getitem__(ptr)
#
#
# C.__getitem__ = __getitem__c
#
#
# def __getitem__l(sequence: L, ptr: Tuple[T, T]) -> L:
#     if is_type(ptr, Tuple[T, T]):
#         batch_ptr, token_ptr = ptr
#         return sequence._replace(data=sequence.data[batch_ptr, token_ptr])
#
#     return super(L, sequence).__getitem__(ptr)
#
#
# L.__getitem__ = __getitem__l
#
#
# def __getitem__p(sequence: P, ptr: Tuple[T, T]) -> P:
#     if is_type(ptr, Tuple[T, T]):
#         batch_ptr, token_ptr = ptr
#         return sequence._replace(data=sequence.data[batch_ptr + sequence.offsets()[token_ptr]])
#
#     return super(P, sequence).__getitem__(ptr)
#
#
# P.__getitem__ = __getitem__p
#
#
# def __getitem__r(sequence: R, ptr: Tuple[T, T]) -> R:
#     if is_type(ptr, Tuple[T, T]):
#         batch_ptr, token_ptr = ptr
#         return sequence._replace(data=sequence.data[batch_ptr, token_ptr])
#
#     return super(R, sequence).__getitem__(ptr)
#
#
# R.__getitem__ = __getitem__r
