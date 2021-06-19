import torch

from hypothesis import strategies as st

MAX_BATCH_SIZE = 24
MAX_TOKEN_SIZE = 128
MAX_EMBEDDING_DIM = 50


@st.composite
def devices(draw):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device('cuda:0')


@st.composite
def batch_sizes(draw, max_value: int = MAX_BATCH_SIZE):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def token_sizes(draw, max_value: int = MAX_TOKEN_SIZE):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def embedding_dims(draw, max_value: int = MAX_EMBEDDING_DIM):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def token_size_lists(draw, max_token_size: int = MAX_TOKEN_SIZE, max_batch_size: int = MAX_BATCH_SIZE):
    return [
        draw(token_sizes(max_value=max_token_size))
        for _ in range(draw(batch_sizes(max_value=max_batch_size)))
    ]
