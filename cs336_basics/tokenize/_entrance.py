import logging

import click

from cs336_basics.common import constants as C, io
from cs336_basics.common.logging import setup_logging
from cs336_basics.tokenize import bpe, pretoken as pre

setup_logging()
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Tokenize the text data"""
    pass


@cli.command()
@click.option("-c", "--config_path", type=str, required=True)
def train_bpe(config_path: str):
    config = io.load_config(config_path)
    logger.info(f"Training BPE with config: {config}")
    vocab, merges = bpe.train_bpe(**config["train_bpe"])

    vocab_for_write = {k: v.decode("utf-8", errors="ignore") for k, v in vocab.items()}
    io.save_dict(vocab_for_write, config["vocab_path"])
    merges_for_write = map(lambda x: b" ".join(x).decode("utf-8", errors="ignore"), merges)
    io.save_sequence(merges_for_write, config["merges_path"])


@cli.command()
@click.option("--config-path", "-c", required=True, help="configuration file path")
def pretokenize(config_path: str):
    cfg: dict = io.load_config(config_path)
    parallel = cfg.pop("parallel", False)
    file = cfg.pop("file")
    num_processes = cfg.pop("num_processes", C.DEFAULT_MAX_NUM_PROCESSES)
    num_processes = max(1, min(num_processes, C.DEFAULT_MAX_NUM_PROCESSES))

    logger.info(
        f"Tokenizing {file} with {num_processes} processes in {'parallel' if parallel else 'serial'} mode, set by {config_path}"  # noqa: E501
    )

    if parallel:
        result = pre.pretoken_and_count_in_parallel(file, num_processes=num_processes, **cfg)
    else:
        with open(file, "rb") as f:
            result = pre.pretoken_and_count(f.read(), **cfg)

    for idx, (token, cnt) in enumerate(result.items()):
        if idx % 100 == 0:
            print(f"{token}: {cnt}")


if __name__ == "__main__":
    cli()
