import logging
import math

from cs336_basics.common import utils

logger = logging.getLogger(__name__)

__all__ = ["lr_schedule"]


def lr_schedule(name: str, lr: float, iter_idx: int = 0, **kwargs) -> float:
    match name:
        case "cosine_annealing":
            return _cosine_annealing(lr, iter_idx, **kwargs)
        case _:
            logger.info(
                f"Learning rate schedule {name} not implemented, using constant learning rate {lr}"  # noqa: E501
            )
            return lr


@utils.require_param(["lr_min", "n_iter_warmup", "n_cosine_annealing"])
def _cosine_annealing(lr: float, iter_idx: int, **kwargs) -> float:
    lr_max = lr
    lr_min = kwargs["lr_min"]
    n_iter_warmup = kwargs["n_iter_warmup"]
    n_cosine_annealing = kwargs["n_cosine_annealing"]

    if iter_idx < n_iter_warmup:
        return (iter_idx / n_iter_warmup) * lr_max
    elif n_iter_warmup <= iter_idx <= n_cosine_annealing:
        return (
            lr_min
            + (lr_max - lr_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (iter_idx - n_iter_warmup)
                    / (n_cosine_annealing - n_iter_warmup)
                )
            )
            / 2
        )
    else:
        return lr_min
