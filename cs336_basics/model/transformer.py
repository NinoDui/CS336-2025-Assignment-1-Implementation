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


class TransformerLM(torch.nn.Module):
    """Transformer Language Model

    A language model whose
        - input: a batched sequence of token ids (B, L)
        - output: the probability of each token in the vocabulary (B, L, V)
    and supports training strategies like:
        - Causal Masking Attention (Autoregressive)
        - RoPE (Rotary Positional Encoding)
        - SwiGLU (Swish-Gated Linear Unit)
        - RMSNorm (Root Mean Square Layer Normalization)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        *,
        device: torch.device | None = None,
        **kwargs,
    ):
        """Transformer Language Model

        Args:
            vocab_size (int):
                The size of the vocabulary
                --> determines the dimensionality of the token emb matrix
            context_length (int):
                The length of the context
                --> determines the dimensionality of the position emb matrix
            d_model (int):
                The input/output dimension of the model
            num_layers (int):
                The number of transformer blocks
            num_heads (int):
                The number of attention heads per transformer block
            d_ff (int):
                The dimension of the feedforward network (SwiGLU)
            device (torch.device | None, optional):
                The device to use for the model
                Default to "CPU"
            **kwargs:
                Additional arguments for the transformer block
                - causal_mask: Whether to apply a causal mask to the attention matrix
                - apply_rope: Whether to apply RoPE to the query and key vectors
                - rope_theta: The theta value for RoPE
                - rope_max_seq_len: The maximum sequence length for RoPE
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device or torch.device(C.DEFAULT_DEVICE)

        self.token_emb = M.Embedding(vocab_size, d_model, device=self.device)
        self.tfm_layers = torch.nn.Sequential(
            *[
                TransformerBlock(
                    d_model, num_heads, d_ff, rope_max_seq_len=context_length, **kwargs
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = M.RMSNorm(d_model, device=self.device)
        self.outptu_emb = M.Linear(d_model, vocab_size, device=self.device)

    def forward(
        self, token_ids: Int[torch.Tensor, "batch_size seq_len"]
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        x = self.token_emb(token_ids)  # (B, L) -> (B, L, D)
        x = self.tfm_layers(x)
        x = self.final_norm(x)
        x = self.outptu_emb(x)  # (B, L, D) -> (B, L, V)

        return x
