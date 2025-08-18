from collections.abc import Callable
import math

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> float | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                iter_idx = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(iter_idx + 1) * grad
                state["t"] = iter_idx + 1

        return loss


if __name__ == "__main__":
    lr_options = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    n_iteration = 10

    for lr in lr_options:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        optimizer = SGD([weights], lr=lr)
        print(f"========== Learning rate: {lr} ==========")
        for _ in range(n_iteration):
            optimizer.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())

            loss.backward()
            optimizer.step()
