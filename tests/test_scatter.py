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

    inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
    index = torch.randint(0, num, (token_size,), device=device)

    actual = rua_scatter.scatter_add(tensor=inputs, index=index)
    excepted = torch_scatter.scatter_add(src=inputs, index=index, dim=0)

    assert_close(actual, excepted)
    assert_grad_close(actual, excepted, inputs=inputs)


@given(
    data=st.data(),
    token_size=token_sizes(),
    num=token_sizes(),
    dim=embedding_dims(),
    device=devices(),
)
def test_scatter_mean(data, token_size, num, dim, device):
    if num > token_size:
        num, token_size = token_size, num

    inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
    index = torch.randint(0, num, (token_size,), device=device)

    actual = rua_scatter.scatter_mean(tensor=inputs, index=index)
    excepted = torch_scatter.scatter_mean(src=inputs, index=index, dim=0)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


@given(
    data=st.data(),
    token_size=token_sizes(),
    num=token_sizes(),
    dim=embedding_dims(),
    device=devices(),
)
def test_scatter_max(data, token_size, num, dim, device):
    if num > token_size:
        num, token_size = token_size, num

    inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
    index = torch.randint(0, num, (token_size,), device=device)

    actual = rua_scatter.scatter_max(tensor=inputs, index=index)
    excepted, _ = torch_scatter.scatter_max(src=inputs, index=index, dim=0)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


@given(
    data=st.data(),
    token_size=token_sizes(),
    num=token_sizes(),
    dim=embedding_dims(),
    device=devices(),
)
def test_scatter_min(data, token_size, num, dim, device):
    if num > token_size:
        num, token_size = token_size, num

    inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
    index = torch.randint(0, num, (token_size,), device=device)

    actual = rua_scatter.scatter_min(tensor=inputs, index=index)
    excepted, _ = torch_scatter.scatter_min(src=inputs, index=index, dim=0)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


@given(
    data=st.data(),
    token_size=token_sizes(),
    num=token_sizes(),
    dim=embedding_dims(),
    device=devices(),
)
def test_scatter_softmax(data, token_size, num, dim, device):
    if num > token_size:
        num, token_size = token_size, num

    inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
    index = torch.randint(0, num, (token_size,), device=device)

    actual = rua_scatter.scatter_softmax(tensor=inputs, index=index)
    excepted = torch_scatter.scatter_softmax(src=inputs, index=index, dim=0)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs)
