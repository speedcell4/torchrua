import torch
import torch_scatter
from hypothesis import given, strategies as st
from torch.testing import assert_close

from tests.strategies import embedding_dims, devices, token_sizes
from tests.utils import assert_grad_close
from torchrua import scatter as rua_scatter


@given(
    data=st.data(),
    token_size=token_sizes(),
    num=token_sizes(),
    dim=embedding_dims(),
    device=devices(),
)
def test_scatter_add(data, token_size, num, dim, device):
    if num > token_size:
        num, token_size = token_size, num

    tensor = torch.randn((token_size, dim), device=device, requires_grad=True)
    index = torch.randint(0, num, (token_size,), device=device)

    prediction = rua_scatter.scatter_add(tensor=tensor, index=index)
    target = torch_scatter.scatter_add(src=tensor, index=index, dim=0)

    assert_close(actual=prediction, expected=target)
    assert_grad_close(actual=prediction, expected=target, inputs=(tensor,))
