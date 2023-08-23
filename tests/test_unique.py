# import torch
# from hypothesis import given
# from torchnyan import BATCH_SIZE
# from torchnyan import FEATURE_DIM
# from torchnyan import TOKEN_SIZE
# from torchnyan import assert_close
# from torchnyan import device
# from torchnyan import sizes
#
# from torchrua import cat_sequence
# from torchrua import pack_sequence
# from torchrua import unique_sequence
#
#
# @given(
#     token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE, FEATURE_DIM),
# )
# def test_unique_catted_sequence(token_sizes):
#     sequence = cat_sequence([
#         torch.tensor(token_size, device=device)
#         for token_size in token_sizes
#     ], device=device)
#
#     unique, inverse, counts = unique_sequence(sequence, device=device)
#
#     assert_close(sequence.data, unique[inverse])
#
#
# @given(
#     token_sizes=sizes(BATCH_SIZE, TOKEN_SIZE, FEATURE_DIM),
# )
# def test_unique_packed_sequence(token_sizes):
#     sequence = pack_sequence([
#         torch.tensor(sequence, device=device)
#         for sequence in token_sizes
#     ], device=device)
#
#     unique, inverse, counts = unique_sequence(sequence, device=device)
#
#     assert_close(sequence.data, unique[inverse])
