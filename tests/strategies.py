import torch

from hypothesis import strategies as st

TINY_BATCH_SIZE = 5
TINY_TOKEN_SIZE = 5
TINY_EMBEDDING_DIM = 25

MAX_BATCH_SIZE = 25
MAX_TOKEN_SIZE = 100
MAX_EMBEDDING_DIM = 25

if torch.cuda.is_available():
    MAX_BATCH_SIZE *= 4
    MAX_TOKEN_SIZE *= 4
    MAX_EMBEDDING_DIM *= 4


@st.composite
def devices(draw):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    _ = torch.empty((1,), device=device)
    return device


@st.composite
def batch_sizes(draw, max_value: int = MAX_BATCH_SIZE):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def batch_size_lists(draw, max_batch_size: int = MAX_BATCH_SIZE):
    return [
        draw(batch_sizes(max_value=max_batch_size))
        for _ in range(draw(batch_sizes(max_value=max_batch_size)))
    ]


@st.composite
def token_sizes(draw, max_value: int = MAX_TOKEN_SIZE):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def token_size_lists(draw, max_token_size: int = MAX_TOKEN_SIZE, max_batch_size: int = MAX_BATCH_SIZE):
    return [
        draw(token_sizes(max_value=max_token_size))
        for _ in range(draw(batch_sizes(max_value=max_batch_size)))
    ]


@st.composite
def embedding_dims(draw, max_value: int = MAX_EMBEDDING_DIM):
    return draw(st.integers(min_value=1, max_value=max_value))
