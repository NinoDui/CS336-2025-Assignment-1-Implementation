import math

import einops
from jaxtyping import Bool, Float
import torch


def softmax(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """
    Softmax function
    """
    if dim is None:
        # softmax over all dimensions
        dim = tuple(range(x.ndim))  # type: ignore[assignment]
    elif dim < 0:
        dim = (x.ndim + dim) % x.ndim

    max_x_on_dim, indices = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_x_on_dim)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    K: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    V: Float[torch.Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[torch.Tensor, "batch_size ... seq_len seq_len"] | None = None,
) -> Float[torch.Tensor, "batch_size ... seq_len d_v"]:
    attn = einops.einsum(Q, K, "... L1 d_k, ... L2 d_k -> ... L1 L2")
    attn = attn / math.sqrt(Q.shape[-1])

    if mask is not None:
        attn = attn.masked_fill(~mask, float("-inf"))

    attn = softmax(attn, dim=-1)
    return attn @ V
