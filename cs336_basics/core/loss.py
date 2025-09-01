from jaxtyping import Float, Int
import torch


def cross_entropy_loss(
    logits: Float[torch.Tensor, "... seq_len vocab_size"],
    targets: Int[torch.Tensor, "... seq_len"],
    reduction: str | None = "mean",
) -> Float[torch.Tensor, ""]:
    """Average cross entropy loss over the batch

    Args:
        logits (Float[torch.Tensor, "... seq_len vocab_size"]):
            Predicted logits of shape (batch_size, seq_len, vocab_size)
        targets (Int[torch.Tensor, "... seq_len"]):
            Target token indices of shape (batch_size, seq_len)
        reduction (str | None, optional):
            Reduction method. Defaults to "mean".

    Returns:
        Float[torch.Tensor, ""]:
            Average cross entropy loss over the batch or the loss per example
    """
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - max_logits

    log_sum = torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    prob_at_true_label = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1).long())  # (B, ..., L, 1)
    loss_in_batch = log_sum - prob_at_true_label

    match reduction:
        case "mean":
            return loss_in_batch.mean()
        case "sum":
            return loss_in_batch.sum()
        case _:
            return loss_in_batch
