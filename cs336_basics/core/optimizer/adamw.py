from collections.abc import Callable, Iterable

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
        pass
