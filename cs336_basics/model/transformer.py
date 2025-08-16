from jaxtyping import Float, Int
import torch

from cs336_basics.common import constants as C
from cs336_basics.core import module as M
from cs336_basics.model import attn


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        device: torch.device | None = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device or torch.device(C.DEFAULT_DEVICE)

        self.attn = attn.MultiHeadSelfAttention(d_model, num_heads, **kwargs)
        self.attn_norm = M.RMSNorm(d_model, device=self.device)

        self.ff = M.SwiGLU(d_model, d_ff=d_ff, device=self.device)
        self.ff_norm = M.RMSNorm(d_model, device=self.device)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        token_positions = token_positions or torch.arange(
            x.shape[-2], device=self.device
        )

        attn_out = self.attn(self.attn_norm(x), token_positions=token_positions)
        x = x + attn_out

        ff_out = self.ff(self.ff_norm(x))
        x = x + ff_out

        return x
