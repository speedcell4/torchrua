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
def draw_device(draw):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    _ = torch.empty((1,), device=device)
    return device


@st.composite
def draw_batch_size(draw, max_value: int = MAX_BATCH_SIZE):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def draw_batch_sizes(draw, max_batch_size: int = MAX_BATCH_SIZE):
    return [
        draw(draw_batch_size(max_value=max_batch_size))
        for _ in range(draw(draw_batch_size(max_value=max_batch_size)))
    ]


@st.composite
def draw_token_size(draw, max_value: int = MAX_TOKEN_SIZE):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def draw_token_sizes(draw, max_token_size: int = MAX_TOKEN_SIZE, max_batch_size: int = MAX_BATCH_SIZE):
    return [
        draw(draw_token_size(max_value=max_token_size))
        for _ in range(draw(draw_batch_size(max_value=max_batch_size)))
    ]


@st.composite
def draw_embedding_dim(draw, max_value: int = MAX_EMBEDDING_DIM):
    return draw(st.integers(min_value=1, max_value=max_value))
