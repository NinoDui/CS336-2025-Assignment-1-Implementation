import torch

from cs336_basics.common import constants as C
from cs336_basics.common import utils
from cs336_basics.core import module as M


class SwiGLU(torch.nn.Module):
    """SwiGLU activation function

    SiLU(x) = x * sigmoid(x) , where sigmoid(x) = 1 / (1 + exp(-x))
    GLU(x, W1, W2) = sigmoid(W1x) * W2x
        - the element-wise product of
          1) a linear transformation passed through a sigmoid function
          2) another linear transformation

    SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else (d_model * 8 // 3)

        self.device = device if device is not None else torch.device(C.DEFAULT_DEVICE)
        self.dtype = dtype if dtype is not None else torch.float32

        self.W1 = M.Linear(
            self.d_model, self.d_ff, device=self.device, dtype=self.dtype
        )
        self.W2 = M.Linear(
            self.d_ff, self.d_model, device=self.device, dtype=self.dtype
        )
        self.W3 = M.Linear(
            self.d_model, self.d_ff, device=self.device, dtype=self.dtype
        )

    @utils.fp32_precision
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin_pass = self.W1(x)  # W1: (d_ff, d_model) x -> (..., d_ff)
        silu_pass = lin_pass * torch.sigmoid(lin_pass)
        lin_pass_2 = self.W3(x)  # W3: (d_ff, d_model) x -> (..., d_ff)
        return self.W2(silu_pass * lin_pass_2)
