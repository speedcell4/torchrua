import torch
from hypothesis import given, settings
from torch.testing import assert_close

from tests.strategy import sizes, TINY_BATCH_SIZE, TINY_TOKEN_SIZE, device
from torchrua.core import major_sizes_to_ptr


@settings(deadline=None)
@given(
    token_sizes=sizes(TINY_BATCH_SIZE, TINY_TOKEN_SIZE),
)
def test_compile_major_sizes_to_ptr(token_sizes):
    token_sizes = torch.tensor(token_sizes, device=device)

    compiled_major_sizes_to_ptr = torch.compile(major_sizes_to_ptr)
    actual1, actual2 = compiled_major_sizes_to_ptr(sizes=token_sizes)

    excepted1, excepted2 = major_sizes_to_ptr(sizes=token_sizes)

    assert_close(actual=actual1, expected=excepted1)
    assert_close(actual=actual2, expected=excepted2)
