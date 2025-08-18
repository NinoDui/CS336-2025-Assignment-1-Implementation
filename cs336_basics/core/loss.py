from jaxtyping import Float, Int
import torch


def cross_entropy_loss(
    logits: Float[torch.Tensor, "... seq_len vocab_size"],
    targets: Int[torch.Tensor, "... seq_len"],
) -> float:
    """Average cross entropy loss over the batch

    Args:
        logits (Float[torch.Tensor, "... seq_len vocab_size"]):
            Predicted logits of shape (batch_size, seq_len, vocab_size)
        targets (Int[torch.Tensor, "... seq_len"]):
            Target token indices of shape (batch_size, seq_len)

    Returns:
        float:
            Average cross entropy loss over the batch
    """
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - max_logits

    log_sum = torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    prob_at_true_label = logits * torch.eye(logits.shape[-1])[targets]  # (B, ..., L, 1)
    loss_in_batch = log_sum - torch.sum(
        prob_at_true_label, dim=-1, keepdim=True
    )  # (B, ..., L, 1)
    return loss_in_batch.mean()
