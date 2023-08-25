# import torch
# from hypothesis import given
# from hypothesis import settings
# from torch import nn
# from torchnyan import FEATURE_DIM
# from torchnyan import TINY_BATCH_SIZE
# from torchnyan import TINY_TOKEN_SIZE
# from torchnyan import assert_close
# from torchnyan import assert_grad_close
# from torchnyan import device
# from torchnyan import sizes
#
# from torchrua import C
# from torchrua import compose_catted_sequences
# from torchrua import pack_sequence
#
#
# @given(
#     token_sizes_batch=sizes(TINY_BATCH_SIZE, TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
#     input_size=sizes(FEATURE_DIM),
#     hidden_size=sizes(FEATURE_DIM),
# )
# @settings(deadline=None)
# def test_compose_catted_sequences(token_sizes_batch, input_size, hidden_size):
#     sequences = [
#         [
#             torch.randn((token_size, input_size), requires_grad=True, device=device)
#             for token_size in token_sizes
#         ]
#         for token_sizes in token_sizes_batch
#     ]
#     inputs = [token for seq in sequences for token in seq]
#     catted_sequences = [cat_sequence(seq, device=device) for seq in sequences]
#     packed_sequences = [pack_sequence(seq, device=device) for seq in sequences]
#
#     rnn = nn.LSTM(
#         input_size=input_size,
#         hidden_size=hidden_size,
#         bidirectional=True, bias=True,
#     ).to(device=device)
#
#     _, (actual, _) = rnn(compose_catted_sequences(*catted_sequences, device=device))
#     actual = actual.transpose(-3, -2).flatten(start_dim=-2)
#
#     expected = []
#     for packed_sequence in packed_sequences:
#         _, (hidden, _) = rnn(packed_sequence)
#         expected.append(hidden.transpose(-3, -2).flatten(start_dim=-2))
#     expected = pack_sequence(expected).data
#
#     assert_close(actual, expected, check_stride=False)
#     assert_grad_close(actual, expected, inputs=inputs)
