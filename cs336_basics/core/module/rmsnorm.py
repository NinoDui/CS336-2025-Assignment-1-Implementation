import torch

from cs336_basics.common import constants as C
from cs336_basics.common import utils


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization,
    - the internal precision must be conversted to fp32 for numerical stability,
      and avoid any overflows in normalization
    - the gain parameter G is initialized to 1.0 and is a learnable parameter
    """

    def __init__(
        self,
        d_model: int,
        *,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.device = device if device is not None else torch.device(C.DEFAULT_DEVICE)
        self.dtype = dtype if dtype is not None else torch.float32

        self.G = torch.nn.Parameter(
            torch.ones(d_model, device=self.device, dtype=self.dtype)
        )

    @utils.fp32_precision
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_x = torch.sum(x**2, dim=-1, keepdim=True) / self.d_model + self.eps
        rms_x = torch.sqrt(rms_x)
        return x / rms_x * self.G
