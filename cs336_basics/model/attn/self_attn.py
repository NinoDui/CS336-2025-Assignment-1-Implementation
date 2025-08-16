import einops
from jaxtyping import Float, Int
import torch

from cs336_basics.common import constants as C
from cs336_basics.core import functions as F, module as M


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        d_q: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        device: torch.device | None = None,
        apply_rope: bool = True,
        causal_mask: bool = False,
        **kwargs,
    ):
        """Multi-head self-attention module

        Args:
            d_model (int):
                Dimensionality of the Transformer block inputs/outputs
            num_heads (int): Number of heads to use
            d_q (int | None, optional):
                Dimensionality of the query vectors. Defaults to d_model // num_heads.
            d_k (int | None, optional):
                Dimensionality of the key vectors. Defaults to d_model // num_heads.
            d_v (int | None, optional):
                Dimensionality of the value vectors. Defaults to d_model // num_heads.
            apply_rope (bool, optional):
                Whether to apply RoPE to the query and key vectors. Defaults to True.
            causal_mask (bool, optional):
                Whether to apply a causal mask to the attention matrix
                Default to False.
                If True, the "true" or "1" in mask means passing through
            rope_theta (float, optional in kwargs):
                The theta value for RoPE. Defaults to 10000.
            rope_max_seq_len (int, optional in kwargs):
                The maximum sequence length for RoPE. Defaults to 2048.
            device (torch.device | None, optional): Defaults to "CPU'
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device or torch.device(C.DEFAULT_DEVICE)

        self._d_q = d_q or d_model // num_heads
        self._d_k = d_k or d_model // num_heads
        self._d_v = d_v or d_model // num_heads

        self.W_q = M.Linear(d_model, self._d_q * num_heads, device=self.device)
        self.W_k = M.Linear(d_model, self._d_k * num_heads, device=self.device)
        self.W_v = M.Linear(d_model, self._d_v * num_heads, device=self.device)
        self.W_o = M.Linear(self._d_v * num_heads, d_model, device=self.device)

        self.apply_rope = apply_rope
        if apply_rope:
            rope_theta = kwargs.get("rope_theta", C.DEFAULT_ROPE_THETA)
            rope_max_seq_len = kwargs.get("rope_max_seq_len", C.DEFAULT_MAX_SEQ_LEN)
            self.emb = M.RotaryPositionalEmbedding(
                rope_theta, self._d_k, rope_max_seq_len, device=self.device
            )
        else:
            self.emb = None  # type: ignore[assignment]

        self.causal_mask = causal_mask

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = einops.rearrange(
            Q, "... seq_len (h d_q) -> ... h seq_len d_q", h=self.num_heads
        )
        K = einops.rearrange(
            K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads
        )
        V = einops.rearrange(
            V, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads
        )

        if self.emb is not None and isinstance(self.emb, M.RotaryPositionalEmbedding):
            if token_positions is None:
                raise ValueError(
                    "token_positions must be provided when apply_rope is True"
                )
            Q = self.emb(Q, token_positions=token_positions)
            K = self.emb(K, token_positions=token_positions)

        if self.causal_mask:
            # True: pass, False: mask
            causal_mask = torch.tril(
                torch.ones(x.shape[-2], x.shape[-2], device=self.device)
            ).bool()
        else:
            causal_mask = None

        o = F.scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        o = einops.rearrange(o, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.W_o(o)

    def __repr__(self):
        params = [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        params += [
            f"positional_emb={self.emb}"
            if self.emb is not None
            else "positional_emb=None"
        ]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def __str__(self):
        return self.__repr__()
