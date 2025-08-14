from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Tensor
import torch

from cs336_basics.common import constants as C


class RotaryPositionalEmbedding(torch.nn.Module):
    """Rotary Positional Embedding (RoPE)
    For a given query token $q^i = W_q x^i$, apply a rotation matrix $R_i$ to the query token, where the matrix will rotate pairs of embedding elements,
    q^i_{2k-1, 2k} as 2D vectors by the angle $\theta_{i,k} = i/\Theta^{(2k-1)/d}$
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
            device (torch.device | None, optional): Device to store the buffer on. Defaults to CPU.
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device(C.DEFAULT_DEVICE)

        # Generate [1, 3, 5, ..., d_k - 1] and repeat each element twice
        self._t_k = torch.arange(1, d_k, 2, device=self.device).repeat_interleave(2)
        self._t_i = torch.arange(1, d_k + 1, device=self.device)
        self._t_theta = self._t_i / torch.pow(self.theta, (2 * self._t_k - 1) / d_k)

        self._cos_theta = torch.cos(self._t_theta)
        self._sin_theta = torch.sin(self._t_theta)

        self.register_buffer("cos_theta", self._cos_theta, persistent=True)
        self.register_buffer("sin_theta", self._sin_theta, persistent=True)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Forward pass of the RoPE module

        Args:
            x (Float[Tensor, "... seq_len d_k"]): Input tensor of shape (..., seq_len, d_k)
            token_positions (Int[Tensor, "... seq_len"]): Token positions of shape (..., seq_len)

        Returns:
            Float[Tensor, "... seq_len d_k"]: Output tensor of shape (..., seq_len, d_k)
        """
        cos_theta_i = self.cos_theta[token_positions]
        sin_theta_i = self.sin_theta[token_positions]

        x_reordered = torch.stack([-x.roll(-1), x], dim=-1).flatten()[:-1]
        x_rotated = x * cos_theta_i + x_reordered * sin_theta_i
        return x_rotated
