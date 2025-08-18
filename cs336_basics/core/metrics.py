from jaxtyping import Float, Int
import torch

from cs336_basics.core import loss


def perplexity(
    logits: Float[torch.Tensor, "... seq_len vocab_size"],
    targets: Int[torch.Tensor, "... seq_len"],
) -> float:
    """Perplexity of the model

    Args:
        logits (Float[torch.Tensor, "... seq_len vocab_size"]):
            Predicted logits of shape (batch_size, seq_len, vocab_size)
        targets (Int[torch.Tensor, "... seq_len"]):
            Target token indices of shape (batch_size, seq_len)

    Returns:
        float:
            Perplexity of the model
    """
    t_loss = loss.cross_entropy_loss(logits, targets, reduction=None)  # (B, ..., L, 1)
    return torch.exp(t_loss.sum(dim=-1))  # type: ignore[union-attr]
