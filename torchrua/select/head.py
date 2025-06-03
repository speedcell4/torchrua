import torch

from torchrua.layout import C, L, P, R


def cat_head(self: C, n: int) -> C:
    data, token_sizes1 = self
    token_sizes0 = torch.full_like(token_sizes1, fill_value=n)

    split_sizes = torch.stack([token_sizes0, token_sizes1 - token_sizes0], dim=-1)
    split_sizes = split_sizes.view(-1).detach().cpu().tolist()

    data = torch.split_with_sizes(data, split_sizes=split_sizes, dim=0)
    data = torch.cat(data[0::2], dim=0)

    return C(data=data, token_sizes=token_sizes0)


C.head = cat_head


def pack_head(self: P, n: int) -> P:
    data, batch_sizes, sorted_indices, unsorted_indices = self

    return P(
        data=data[:batch_sizes[0].detach().cpu().item() * n],
        batch_sizes=batch_sizes[:n],
        sorted_indices=sorted_indices,
        unsorted_indices=unsorted_indices,
    )


P.head = pack_head


def left_head(self: L, n: int) -> L:
    data, token_sizes = self

    return L(
        data=data[:, :n],
        token_sizes=torch.full_like(token_sizes, n),
    )


L.head = left_head


def right_head(self: R, n: int) -> R:
    data, token_sizes2 = self

    b, t, *_ = data.size()
    token_sizes0 = torch.full_like(token_sizes2, fill_value=t)
    token_sizes1 = torch.full_like(token_sizes2, fill_value=n)

    split_sizes = torch.stack([token_sizes0 - token_sizes2, token_sizes1, token_sizes2 - n], dim=-1)
    split_sizes = split_sizes.view(-1).detach().cpu().tolist()

    data = torch.split_with_sizes(data.flatten(end_dim=1), split_sizes=split_sizes, dim=0)
    data = torch.stack(data[1::3], dim=0)

    return R(
        data=data,
        token_sizes=token_sizes1,
    )


R.head = right_head
