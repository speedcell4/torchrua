import torch
from hypothesis import strategies as st

TINY_BATCH_SIZE = 5
TINY_TOKEN_SIZE = 10
TINY_EMBEDDING_DIM = 12

if torch.cuda.is_available():
    BATCH_SIZE = 50
    TOKEN_SIZE = 80
    EMBEDDING_DIM = 100
else:
    BATCH_SIZE = 30
    TOKEN_SIZE = 50
    EMBEDDING_DIM = 60

if torch.cuda.is_available():
    torch.empty((1,), device=torch.device('cuda:0'))


@st.composite
def devices(draw):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    return device


@st.composite
def sizes(draw, *size: int, min_size: int = 1):
    max_size, *size = size

    if len(size) == 0:
        return draw(st.integers(min_value=min_size, max_value=max_size))
    else:
        return [
            draw(sizes(*size, min_size=min_size))
            for _ in range(draw(st.integers(min_value=min_size, max_value=max_size)))
        ]
