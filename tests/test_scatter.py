import torch
from hypothesis import given, strategies as st
from torch.testing import assert_close

from tests.strategies import embedding_dims, devices, token_sizes, TINY_TOKEN_SIZE
from tests.utils import assert_grad_close
from torchrua import scatter as rua_scatter

try:
    import torch_scatter


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
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_add(tensor=inputs, index=index1)
        excepted = torch_scatter.scatter_add(src=inputs, index=index2, dim=0)

        assert_close(actual, excepted)
        assert_grad_close(actual, excepted, inputs=inputs)


    @given(
        data=st.data(),
        token_size=token_sizes(),
        num=token_sizes(),
        dim=embedding_dims(),
        device=devices(),
    )
    def test_scatter_mul(data, token_size, num, dim, device):
        if num > token_size:
            num, token_size = token_size, num

        inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_mul(tensor=inputs, index=index1)
        excepted = torch_scatter.scatter_mul(src=inputs, index=index2, dim=0)

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
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_mean(tensor=inputs, index=index1)
        excepted = torch_scatter.scatter_mean(src=inputs, index=index2, dim=0)

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
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_max(tensor=inputs, index=index1)
        excepted, _ = torch_scatter.scatter_max(src=inputs, index=index2, dim=0)

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
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_min(tensor=inputs, index=index1)
        excepted, _ = torch_scatter.scatter_min(src=inputs, index=index2, dim=0)

        assert_close(actual=actual, expected=excepted)
        assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


    @given(
        data=st.data(),
        token_size=token_sizes(max_value=TINY_TOKEN_SIZE),
        num=token_sizes(max_value=TINY_TOKEN_SIZE),
        dim=embedding_dims(),
        device=devices(),
    )
    def test_scatter_logsumexp(data, token_size, num, dim, device):
        if num > token_size:
            num, token_size = token_size, num

        inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_logsumexp(tensor=inputs, index=index1)
        excepted = torch_scatter.scatter_logsumexp(src=inputs, index=index2, dim=0)

        mask = ~torch.isinf(excepted)
        actual = torch.masked_select(actual, mask=mask)
        excepted = torch.masked_select(excepted, mask=mask)

        assert_close(actual=actual, expected=excepted)
        assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


    @given(
        data=st.data(),
        token_size=token_sizes(max_value=TINY_TOKEN_SIZE),
        num=token_sizes(max_value=TINY_TOKEN_SIZE),
        dim=embedding_dims(),
        device=devices(),
    )
    def test_scatter_softmax(data, token_size, num, dim, device):
        if num > token_size:
            num, token_size = token_size, num

        inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_softmax(tensor=inputs, index=index1)
        excepted = torch_scatter.scatter_softmax(src=inputs, index=index2, dim=0)

        assert_close(actual=actual, expected=excepted)
        assert_grad_close(actual=actual, expected=excepted, inputs=inputs)


    @given(
        data=st.data(),
        token_size=token_sizes(max_value=TINY_TOKEN_SIZE),
        num=token_sizes(max_value=TINY_TOKEN_SIZE),
        dim=embedding_dims(),
        device=devices(),
    )
    def test_scatter_log_softmax(data, token_size, num, dim, device):
        if num > token_size:
            num, token_size = token_size, num

        inputs = torch.randn((token_size, dim), device=device, requires_grad=True)
        index1 = torch.randint(0, num, (token_size,), device=device)
        index2 = index1[:, None].expand_as(inputs)

        actual = rua_scatter.scatter_log_softmax(tensor=inputs, index=index1)
        excepted = torch_scatter.scatter_log_softmax(src=inputs, index=index2, dim=0)

        assert_close(actual=actual, expected=excepted)
        assert_grad_close(actual=actual, expected=excepted, inputs=inputs)

except ModuleNotFoundError:
    pass
