from collections.abc import Callable, Iterable
import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        lambda_weight_decay: float = 1e-2,
        **kwargs,
    ):
        """AdamW optimizer

        Args:
            params (Iterable[torch.nn.Parameter]):
                Parameters to optimize
            lr (float):
                Learning rate
            betas (tuple[float, float], optional):
                Coefficients used for computing 1st/2nd moment estimates.
                Defaults to (0.9, 0.999).
            eps (float, optional):
                Term added to the denominator to improve numerical stability.
                Defaults to 1e-8.
            lambda_weight_decay (float, optional):
                Lambda value for weight decay.
                Defaults to 1e-2.
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps, lambda_weight_decay=lambda_weight_decay
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            lambda_weight_decay = group["lambda_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 0) + 1

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2
                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * lambda_weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t

        return loss
