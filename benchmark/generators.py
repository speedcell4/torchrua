import torch

if torch.cuda.is_available():
    MAX_BATCH_SIZE = 120
    TINY_BATCH_SIZE = 24

    MAX_TOKEN_SIZE = 512
    MAX_EMBEDDING_DIM = 100

else:
    MAX_BATCH_SIZE = 24
    TINY_BATCH_SIZE = 24

    MAX_TOKEN_SIZE = 128
    MAX_EMBEDDING_DIM = 50


def draw_devices():
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    _ = torch.empty((1,), device=device)
    return device


def draw_batch_sizes(max_batch_size: int = MAX_BATCH_SIZE):
    return torch.randint(1, max_batch_size + 1, (1,)).item()


def draw_batch_size_lists(max_batch_size: int = TINY_BATCH_SIZE):
    return [
        draw_batch_sizes(max_batch_size=max_batch_size)
        for _ in range(draw_batch_sizes(max_batch_size=max_batch_size))
    ]


def draw_token_sizes(max_token_size: int = MAX_TOKEN_SIZE):
    return torch.randint(1, max_token_size + 1, (1,)).item()


def draw_token_size_lists(max_token_size: int = MAX_TOKEN_SIZE, max_batch_size: int = MAX_BATCH_SIZE):
    return [
        draw_token_sizes(max_token_size=max_token_size)
        for _ in range(draw_batch_sizes(max_batch_size=max_batch_size))
    ]


def draw_embedding_dims(max_dim: int = MAX_EMBEDDING_DIM):
    return torch.randint(1, max_dim + 1, (1,)).item()