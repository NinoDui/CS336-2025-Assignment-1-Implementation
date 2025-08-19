import torch

from cs336_basics.common import constants as C, types as T


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: T.FileType,
):
    """
    Save the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The iteration number.
        out (dst_type): The destination to save the checkpoint.
    """
    torch.save(
        {
            C.DEFAULT_MODEL_STATE_DICT_KEY: model.state_dict(),
            C.DEFAULT_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
            C.DEFAULT_ITERATION_KEY: iteration,
        },
        out,
    )


def load_checkpoint(
    src: T.FileType, model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Load the model and optimizer state from a file.

    Args:
        src (dst_type): The source to load the checkpoint from.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint[C.DEFAULT_MODEL_STATE_DICT_KEY])
    optimizer.load_state_dict(checkpoint[C.DEFAULT_OPTIMIZER_STATE_DICT_KEY])
    return checkpoint[C.DEFAULT_ITERATION_KEY]
