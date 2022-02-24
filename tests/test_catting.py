import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_sequence as torch_pack_sequence
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence

from tests.strategies import draw_token_sizes, draw_embedding_dim, draw_device
from tests.utils import assert_grad_close, assert_catted_sequence_close
from torchrua.catting import cat_sequence, cat_padded_sequence, cat_packed_sequence


@given(
    token_sizes=draw_token_sizes(),
    dim=draw_embedding_dim(),
    device=draw_device(),
)
def test_cat_packed_sequence(token_sizes, dim, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    packed_sequence = torch_pack_sequence(inputs, enforce_sorted=False)

    actual = cat_sequence(inputs, device=device)
    expected = cat_packed_sequence(packed_sequence, device=device)

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)


@given(
    token_sizes=draw_token_sizes(),
    dim=draw_embedding_dim(),
    batch_first=st.booleans(),
    device=draw_device(),
)
def test_cat_padded_sequence(token_sizes, dim, batch_first, device):
    inputs = [
        torch.randn((token_size, dim), device=device, requires_grad=True)
        for token_size in token_sizes
    ]
    sequence = torch_pad_sequence(inputs, batch_first=batch_first)
    token_sizes = torch.tensor(token_sizes, device=device)

    actual = cat_sequence(inputs, device=device)
    expected = cat_padded_sequence(sequence, token_sizes, batch_first=batch_first, device=device)

    assert_catted_sequence_close(actual=actual, expected=expected)
    assert_grad_close(actual=actual.data, expected=expected.data, inputs=inputs)
