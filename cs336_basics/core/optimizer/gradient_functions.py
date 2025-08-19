from collections.abc import Iterable

import torch

from cs336_basics.core import functions as F


def gradient_clipping(
    params: Iterable[torch.nn.Parameter], max_l2_norm: float, *, eps: float = 1e-6
):
    gradients = [p.grad for p in params if p.grad is not None]
    if gradients is None or len(gradients) == 0:
        return

    total_norm = F.l2_norm(torch.cat([g.view(-1) for g in gradients]))
    if total_norm > max_l2_norm:
        scaling_factor = max_l2_norm / (total_norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scaling_factor)
