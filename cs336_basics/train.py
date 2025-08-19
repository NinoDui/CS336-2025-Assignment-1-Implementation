from collections.abc import Generator
import pprint

import click
import numpy as np
import torch
import wandb

from cs336_basics import pipeline as pl
from cs336_basics.common import constants as C, io
from cs336_basics.core import loss as L, optimizer as opt
from cs336_basics.model import transformer as tfm


@click.command()
# TODO(nino): update to Hydra for compositional configuration
@click.option(
    "-c", "--config", type=click.Path(exists=True), help="Path to the config file"
)
def start_training(config: str):
    cfg = io.load_config(config)
    pprint.pprint(cfg)

    # General configuration
    device = cfg.get(
        "device", "cuda:0" if torch.cuda.is_available() else C.DEFAULT_DEVICE
    )
    resume = cfg.get("resume", False)
    enable_gradient_clipping = cfg.get("enable_gradient_clipping", True)

    # Initialize the model
    model = tfm.TransformerLM(**cfg["model"])
    optimizer = opt.AdamW(model.parameters(), **cfg["optimizer"])

    # Resume or train from the scratch
    if resume:
        ckpt_path = cfg.get("ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("Checkpoint path is required for resume training")
        start_iteration = pl.load_checkpoint(ckpt_path, model, optimizer)
    else:
        start_iteration = 0

    model = model.to(device)
    optimizer = optimizer.to(device)

    # Initialize the weight & bias logger
    wb_logger = wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["name"],
        config=cfg,
    )

    # load the dataset
    ds_train = np.load(cfg["dataset"]["src_path"], mmap_mode="r")

    for iter_idx in range(start_iteration, cfg["max_iterations"]):
        pair = pl.data_loading(ds_train, **cfg["dataset"])
        if isinstance(pair, tuple):
            x, y = pair
        elif isinstance(pair, Generator):
            x, y = next(pair)
        else:
            raise ValueError(f"Invalid data loading type: {type(pair)}")
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = L.cross_entropy_loss(logits, y)

        # Backward pass
        loss.backward()  # type: ignore[union-attr]

        # Gradient clipping
        if enable_gradient_clipping:
            opt.gradient_clipping(model.parameters(), **cfg["gradient_clipping"])

        # Update the model
        optimizer.step()

        # Update the learning rate scheduler
        # TODO(nino): optimize the impl of lr_schedule, it's bare to broadcast the lr to parameter groups in this way     # noqa: E501
        lr = opt.lr_schedule(iter_idx=iter_idx, **cfg["lr_scheduler"])
        for p in optimizer.param_groups():
            p["lr"] = lr

        # Log the loss
        wb_logger.log({"loss": loss.item()})


if __name__ == "__main__":
    start_training()
