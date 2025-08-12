import math

import torch

from cs336_basics.common import constants as C


class Linear(torch.nn.Module):
    """Linear Layer reimplemented following:
        1. Weights only, without bias (as in the most of current LLM models)
        2. Weight is initialized with truncated normal distribution:
            - mean: 0
            - std: sqrt(2 / (in_features + out_features))
            - truncated range: (-3 * std, 3 * std)

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        device (torch.device, optional): Device to use for the module. Defaults to cpu. # noqa: E501
        dtype (torch.dtype, optional): Data type to use for the module. Defaults to float32. # noqa: E501
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self._init_mean = 0
        self._init_std = math.sqrt(2 / (in_features + out_features))
        self._init_trunc_coe = (-3, 3)

        self.device = device if device is not None else torch.device(C.DEFAULT_DEVICE)
        self.dtype = dtype if dtype is not None else torch.float32

        self.W = torch.nn.Parameter(
            torch.zeros(out_features, in_features, device=self.device, dtype=self.dtype)
        )
        torch.nn.init.trunc_normal_(
            self.W,
            mean=self._init_mean,
            std=self._init_std,
            a=self._init_trunc_coe[0] * self._init_std,
            b=self._init_trunc_coe[1] * self._init_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
