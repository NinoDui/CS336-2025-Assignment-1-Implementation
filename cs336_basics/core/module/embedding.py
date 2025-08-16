import torch

from cs336_basics.common import constants as C


class Embedding(torch.nn.Module):
    """Embedding Layer reimplemented following:
    1. Vocabulary is initialized with truncated normal distribution:
        - mean: 0
        - std: 1
        - truncated range: (-3, 3)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.device = device if device is not None else torch.device(C.DEFAULT_DEVICE)
        self.dtype = dtype if dtype is not None else torch.float32

        self._init_mean = 0
        self._init_std = 1
        self._init_trunc_coe = (-3, 3)

        self.vocab = torch.nn.Parameter(
            torch.zeros(
                self.num_embeddings,
                self.embedding_dim,
                dtype=self.dtype,
                device=self.device,
            )
        )
        torch.nn.init.trunc_normal_(
            self.vocab,
            mean=self._init_mean,
            std=self._init_std,
            a=self._init_trunc_coe[0] * self._init_std,
            b=self._init_trunc_coe[1] * self._init_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """For each position (i,j) in X, take the embedding at index X[i,j] from vocab,
        and arrange the results in a tensor of shape X.shape + voc.shape[1:]

        Args:
            x (torch.Tensor): input Tensor of shape (..., sequence_length)

        Returns:
            torch.Tensor: output Tensor of shape (..., sequence_length, embedding_dim)
        """
        return self.vocab[x]

    def __repr__(self):
        params = [
            f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("__")
        ]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def __str__(self):
        return self.__repr__()
