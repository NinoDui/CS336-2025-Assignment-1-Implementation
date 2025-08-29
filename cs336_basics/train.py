import os
import pprint

import click
import numpy as np
import torch

from cs336_basics import pipeline as pl
from cs336_basics.common import io
from cs336_basics.core import loss as L, optimizer as opt
from cs336_basics.model import transformer as tfm


# TODO(nino): update to Hydra for compositional configuration
# TODO(nino): rewrite the validation step
# TODO(nino): break the loop into pytorch-lightning style func pieces, a long loop is ugly... # noqa: E501
# TODO(nino): optimize the impl of lr_schedule, it's bare to broadcast the lr to parameter groups directly in this impl     # noqa: E501
@click.command()
@click.option("-c", "--config", type=click.Path(exists=True), help="Path to the config file")
def start_training(config: str):
    cfg = io.load_config(config)
    pprint.pprint(cfg)

    # General configuration
    device = os.environ.get("DEVICE", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    resume = cfg.get("resume", False)
    enable_gradient_clipping = cfg.get("enable_gradient_clipping", True)
    # valid_interval = cfg.get("valid_interval", 50)
    save_interval = cfg.get("save_interval", 100)

    # Initialize the model
    model = tfm.TransformerLM(**cfg["model"])
    optimizer = opt.AdamW(model.parameters(), **cfg["optimizer"])

    # Resume from the checkpoint
    if resume:
        ckpt_path = cfg.get("resume_ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("Checkpoint path is required for resume training")
        start_iteration = pl.load_checkpoint(ckpt_path, model, optimizer)
    else:
        start_iteration = 0

    model = model.to(device)

    # Initialize the weight & bias logger
    # wb_logger = wandb.init(
    #     project=cfg["wandb"]["project"],
    #     name=cfg["wandb"]["name"],
    #     config=cfg,
    # )

    # load the dataset
    ds_train = np.load(cfg["dataset"]["src_path"], mmap_mode="r")

    for iter_idx in range(start_iteration, cfg["max_iterations"]):
        x, y = pl.data_loading(ds_train, **cfg["dataset"])
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = L.cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if enable_gradient_clipping:
            opt.gradient_clipping(model.parameters(), **cfg["gradient_clipping"])

        # Update the learning rate scheduler
        lr = opt.lr_schedule(iter_idx=iter_idx, **cfg["lr_scheduler"])
        for p in optimizer.param_groups():
            p["lr"] = lr

        # Update the model
        optimizer.step()

        # Log the loss to Weights & Biases
        print(f"iter_idx: {iter_idx}, loss: {loss.item()}, lr: {lr}")
        # wb_logger.log({"loss": loss.item()})

        # # Validate the model
        # if iter_idx % valid_interval == 0:
        #     pl.validate(model, optimizer, **cfg["dataset"])

        # Save the model
        if iter_idx % save_interval == 0:
            pl.save_checkpoint(model, optimizer, iter_idx, **cfg["checkpoint"])


if __name__ == "__main__":
    start_training()
