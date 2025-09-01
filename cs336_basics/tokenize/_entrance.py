import logging
import os

import click
import numpy as np

from cs336_basics.common import constants as C, io
from cs336_basics.tokenize import bpe, pretoken as pre, tokenizer as tk

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

    def _decode(x: bytes) -> str:
        try:
            return x.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return x.decode("latin-1", errors="ignore")

    vocab_for_write = {k: _decode(v) for k, v in vocab.items()}
    io.save_dict(vocab_for_write, config["vocab_path"])

    merges_for_write = map(lambda x: b" ".join(x), merges)
    io.save_sequence(merges_for_write, config["merges_path"], save_bytes=True)


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


@cli.command()
@click.option("-c", "--config_path", type=str, required=True)
def tokenize(config_path: str):
    cfg: dict = io.load_config(config_path)
    logger.info(f"Tokenizing with config: {cfg}")

    tokenizer = tk.Tokenizer.from_file(**cfg["tokenizer"])
    token_ids = []
    with open(cfg["input_file"], "rb") as f:
        buffer = b""
        while True:
            content = f.read(C.DEFAULT_MAX_CHUNK_SIZE)
            if not content or len(content) == 0:
                break

            buffer += content
            last_space_pos = buffer.rfind(b" ")
            if last_space_pos != -1:
                valid_content = buffer[:last_space_pos]
                buffer = buffer[last_space_pos + 1 :]
            else:
                valid_content = buffer
                buffer = b""

            tokens = tokenizer.encode(valid_content.decode("utf-8", errors="replace"))
            token_ids.extend(tokens)

            logger.info(f"Tokenized {len(valid_content)} bytes into {len(tokens)} tokens")

        np_token_ids = np.array(token_ids, dtype=np.int32)

    output_folder = cfg.get("output_folder", os.path.dirname(cfg["input_file"]))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, os.path.basename(cfg["input_file"]).replace(".txt", ".npy"))
    np.save(output_file, np_token_ids)


if __name__ == "__main__":
    cli()
