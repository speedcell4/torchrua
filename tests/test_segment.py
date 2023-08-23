# import torch
# from hypothesis import given
# from hypothesis import strategies as st
# from torch import Tensor
# from torch.testing import assert_close
# from torchnyan import BATCH_SIZE
# from torchnyan import FEATURE_DIM
# from torchnyan import TOKEN_SIZE
# from torchnyan import assert_catted_sequence_close
# from torchnyan import assert_grad_close
# from torchnyan import assert_packed_sequence_close
# from torchnyan import device
# from torchnyan import sizes
#
# from torchrua import pad_sequence
# from torchrua import segment_sequence
# from torchrua.catting import cat_sequence
# from torchrua.packing import pack_sequence
# from torchrua.reduce import segment_max
# from torchrua.reduce import segment_mean
# from torchrua.reduce import segment_min
# from torchrua.reduce import segment_sum
#
#
# def reduce_mean(x: Tensor) -> Tensor:
#     return x.mean(dim=0)
#
#
# def reduce_sum(x: Tensor) -> Tensor:
#     return x.sum(dim=0)
#
#
# def reduce_max(x: Tensor) -> Tensor:
#     return x.max(dim=0).values
#
#
# def reduce_min(x: Tensor) -> Tensor:
#     return x.min(dim=0).values
#
#
# def raw_segment(tensor, durations, reduce_fn):
#     expected = []
#
#     for index, duration in enumerate(durations):
#         start, end, seq = 0, 0, []
#         for size in duration:
#             start, end = end, end + size
#             seq.append(reduce_fn(tensor[index][start:end]))
#
#         seq = torch.stack(seq, dim=0)
#         expected.append(seq)
#
#     return expected
#
#
# @given(
#     token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
#     dim=sizes(FEATURE_DIM),
#     reduce_segment=st.sampled_from([
#         (reduce_mean, segment_mean),
#         (reduce_sum, segment_sum),
#         (reduce_max, segment_max),
#         (reduce_min, segment_min),
#     ]),
# )
# def test_segment_catted_sequence(token_sizes, dim, reduce_segment):
#     durations = [
#         torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
#         for token_size in token_sizes
#     ]
#
#     tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)
#
#     reduce_fn, segment_fn = reduce_segment
#
#     actual, _, _ = segment_sequence(
#         tensor=tensor, reduce_fn=segment_fn, keep=False,
#         sizes=cat_sequence(sequence=durations, device=device),
#     )
#
#     expected = cat_sequence(raw_segment(tensor, durations, reduce_fn))
#
#     assert_catted_sequence_close(actual=actual, expected=expected)
#     assert_grad_close(actual=actual.data, expected=expected.data, inputs=tensor)
#
#
# @given(
#     token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
#     dim=sizes(FEATURE_DIM),
#     reduce_segment=st.sampled_from([
#         (reduce_mean, segment_mean),
#         (reduce_sum, segment_sum),
#         (reduce_max, segment_max),
#         (reduce_min, segment_min),
#     ]),
# )
# def test_segment_packed_sequence(token_sizes, dim, reduce_segment):
#     durations = [
#         torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
#         for token_size in token_sizes
#     ]
#
#     tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)
#
#     reduce_fn, segment_fn = reduce_segment
#
#     actual, _, _ = segment_sequence(
#         tensor=tensor, reduce_fn=segment_fn, keep=False,
#         sizes=pack_sequence(sequence=durations, device=device),
#     )
#
#     expected = pack_sequence(raw_segment(tensor, durations, reduce_fn))
#
#     assert_packed_sequence_close(actual=actual, expected=expected)
#     assert_grad_close(actual=actual.data, expected=expected.data, inputs=tensor)
#
#
# @given(
#     token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
#     dim=sizes(FEATURE_DIM),
#     reduce_segment=st.sampled_from([
#         (reduce_mean, segment_mean),
#         (reduce_sum, segment_sum),
#         (reduce_max, segment_max),
#         (reduce_min, segment_min),
#     ]),
# )
# def test_segment_catted_sequence_and_keep(token_sizes, dim, reduce_segment):
#     durations = [
#         torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
#         for token_size in token_sizes
#     ]
#
#     tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)
#
#     reduce_fn, segment_fn = reduce_segment
#
#     actual, mask, _ = segment_sequence(
#         tensor=tensor, reduce_fn=segment_fn, keep=True,
#         sizes=cat_sequence(sequence=durations, device=device),
#     )
#
#     expected, _ = pad_sequence(raw_segment(tensor, durations, reduce_fn))
#
#     assert not torch.isinf(actual).any().item()
#     assert not torch.isnan(actual).any().item()
#
#     assert_close(actual=actual[mask], expected=expected[mask])
#     assert_grad_close(actual=actual, expected=expected, inputs=tensor)
#
#
# @given(
#     token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE),
#     dim=sizes(FEATURE_DIM),
#     reduce_segment=st.sampled_from([
#         (reduce_mean, segment_mean),
#         (reduce_sum, segment_sum),
#         (reduce_max, segment_max),
#         (reduce_min, segment_min),
#     ]),
# )
# def test_segment_packed_sequence_and_keep(token_sizes, dim, reduce_segment):
#     durations = [
#         torch.unique(torch.randint(token_size, (token_size,), device=device), return_counts=True)[1]
#         for token_size in token_sizes
#     ]
#
#     tensor = torch.randn((len(token_sizes), max(token_sizes), dim), device=device, requires_grad=True)
#
#     reduce_fn, segment_fn = reduce_segment
#
#     actual, mask, _ = segment_sequence(
#         tensor=tensor, reduce_fn=segment_fn, keep=True,
#         sizes=pack_sequence(sequence=durations, device=device),
#     )
#
#     expected, _ = pad_sequence(raw_segment(tensor, durations, reduce_fn))
#
#     assert not torch.isinf(actual).any().item()
#     assert not torch.isnan(actual).any().item()
#
#     assert_close(actual=actual[mask], expected=expected[mask])
#     assert_grad_close(actual=actual, expected=expected, inputs=tensor)
