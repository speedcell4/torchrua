import torch
from hypothesis import strategies as st

TINY_BATCH_SIZE = 5
TINY_TOKEN_SIZE = 11
TINY_EMBEDDING_DIM = 13

if torch.cuda.is_available():
    BATCH_SIZE = 53
    TOKEN_SIZE = 83
    EMBEDDING_DIM = 107
else:
    BATCH_SIZE = 37
    TOKEN_SIZE = 53
    EMBEDDING_DIM = 61

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

torch.empty((1,), device=device)


@st.composite
def sizes(draw, *size: int, min_size: int = 1):
    max_size, *size = size
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    if len(size) == 0:
        return n
    else:
        return draw(st.lists(sizes(*size), min_size=n, max_size=n))
