import torch

from hypothesis import strategies as st


@st.composite
def devices(draw):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device('cuda:0')


@st.composite
def batch_sizes(draw, min_value: int = 1, max_value: int = 13):
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def token_sizes(draw, min_value: int = 1, max_value: int = 17):
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def embedding_dims(draw, min_value: int = 1, max_value: int = 19):
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def token_size_lists(draw):
    return [draw(token_sizes()) for _ in range(draw(batch_sizes()))]
