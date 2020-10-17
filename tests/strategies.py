from typing import List

import torch
from hypothesis import strategies as st

RTOL = 1e-5
ATOL = 1e-5


@st.composite
def devices(draw):
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


@st.composite
def batch_size_integer(draw, max_value: int = 7):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def max_sentence_length_integer(draw, max_value: int = 11):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def embedding_dim_integer(draw, max_value: int = 13):
    return draw(st.integers(min_value=1, max_value=max_value))


@st.composite
def list_of_sentence_lengths(
        draw, batch_size: int = None, total_length: int = None):
    if batch_size is None:
        batch_size = draw(batch_size_integer())
    if total_length is None:
        total_length = draw(max_sentence_length_integer())
    return torch.randint(0, total_length, (batch_size,), dtype=torch.long, device=draw(devices())) + 1


@st.composite
def list_of_sentences(
        draw, embedding_dim: int = None, sentence_lengths: List[int] = None, *, return_lengths: bool = False):
    if embedding_dim is None:
        embedding_dim = draw(embedding_dim_integer())
    if sentence_lengths is None:
        sentence_lengths = draw(list_of_sentence_lengths()).detach().cpu().tolist()

    sentences = [
        torch.randn((length, embedding_dim), dtype=torch.float32, device=draw(devices()))
        for index, length in enumerate(sentence_lengths)
    ]
    if not return_lengths:
        return sentences
    return sentences, sentence_lengths


@st.composite
def list_of_homo_lists_of_sentences(
        draw, num: int = None, embedding_dim: int = None, sentence_lengths: List[int] = None):
    if num is None:
        num = draw(batch_size_integer())
    if embedding_dim is None:
        embedding_dim = draw(embedding_dim_integer())
    if sentence_lengths is None:
        sentence_lengths = draw(list_of_sentence_lengths())

    return [
        draw(list_of_sentences(embedding_dim, sentence_lengths))
        for _ in range(num)
    ]
