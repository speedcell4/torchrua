import torch
from hypothesis import strategies as st

TINY_BATCH_SIZE = 5
TINY_TOKEN_SIZE = 7
TINY_EMBEDDING_DIM = 11

if torch.cuda.is_available():
    BATCH_SIZE = 47
    TOKEN_SIZE = 211
    EMBEDDING_DIM = 419
else:
    BATCH_SIZE = 13
    TOKEN_SIZE = 53
    EMBEDDING_DIM = 101


@st.composite
def devices(draw):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    _ = torch.empty((1,), device=device)
    return device


@st.composite
def sizes(draw, *shape: int, min_size: int = 1):
    max_size, *shape = shape

    if len(shape) == 0:
        return draw(st.integers(min_value=min_size, max_value=max_size))
    else:
        return [
            draw(sizes(*shape, min_size=min_size))
            for _ in range(draw(st.integers(min_value=min_size, max_value=max_size)))
        ]
