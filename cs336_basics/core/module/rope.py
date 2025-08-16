from typing import TypeAlias

import einops
from jaxtyping import Float, Int
import torch

from cs336_basics.common import constants as C

Tensor: TypeAlias = torch.Tensor


class RotaryPositionalEmbedding(torch.nn.Module):
    """Rotary Positional Embedding (RoPE)
    For a given query token $q^i = W_q x^i$,
    apply a rotation matrix $R_i$ to the query token,
    where the matrix will rotate pairs of embedding elements,
    q^i_{2k-1, 2k} as 2D vectors by the angle $\theta_{i,k} = i/\\Theta^{(2k-1)/d}$
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        *,
        device: torch.device | None = None,
        **kwargs,
    ):
        """Initialization of the RoPE module

        Args:
            theta (float): Theta value of the RoPE
            d_k (int): dimension of the query and key vectors
            max_seq_len (int): Maximum sequence length that will be used
            device (torch.device | None, optional): Defaults to CPU.
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device(C.DEFAULT_DEVICE)

        self._t_k = torch.arange(0, d_k, 2, device=self.device).repeat_interleave(2)
        self._t_i = torch.arange(0, self.max_seq_len, device=self.device)
        self._t_i = einops.repeat(self._t_i, "max_seq_len -> max_seq_len d_k", d_k=d_k)
        self._t_theta = self._t_i / torch.pow(self.theta, self._t_k / d_k)

        self._cos_theta = torch.cos(self._t_theta)
        self._sin_theta = torch.sin(self._t_theta)

        self.register_buffer("cos_theta_buf", self._cos_theta, persistent=True)
        self.register_buffer("sin_theta_buf", self._sin_theta, persistent=True)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Forward pass of the RoPE module

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)

        Returns:
            Float[Tensor, "... seq_len d_k"]: Output tensor of shape (..., seq_len, d_k)
        """
        cos_theta = self.cos_theta_buf[token_positions]
        sin_theta = self.sin_theta_buf[token_positions]

        # given X of shape (..., seq_len, d), reorder the last dimension:
        # (x0, x1, x2, ..., x_n-1)
        # -> (x0, x1), (x2, x3), ..., (x_n-2, x_n-1)
        # -> (-x1, x0), (-x3, x2), ..., (-x_n-1, x_n-2)
        # -> (-x1, x0, -x3, x2, ..., -x_n-1, x_n-2)
        pairs = einops.rearrange(x, "... seq_len (d_k n) -> ... seq_len d_k n", n=2)
        pairs = torch.stack([-pairs[..., -1], pairs[..., 0]], dim=-1)
        x_reordered = einops.rearrange(
            pairs, "... seq_len d_k n -> ... seq_len (d_k n)", n=2
        )

        return x * cos_theta + x_reordered * sin_theta

    def __repr__(self):
        params = [
            f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("__")
        ]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def __str__(self):
        return self.__repr__()
