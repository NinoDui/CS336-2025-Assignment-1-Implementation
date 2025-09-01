import concurrent.futures as cf
import logging
import os

import click
import numpy as np

from cs336_basics.common import constants as C, io
from cs336_basics.tokenization import bpe, pretoken as pre, tokenizer as tk

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

    output_folder = cfg.get("output_folder", os.path.dirname(cfg["input_file"]))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def reduce(output_folder: str):
        import glob

        token_files = glob.glob(os.path.join(output_folder, "*.npy"))
        token_files = sorted(token_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        token_ids = np.concatenate([np.load(file) for file in token_files])
        np.save(os.path.join(output_folder, "token_all.npy"), token_ids)
        logger.info(f"Merged {len(token_files)} files into {token_ids.shape} tokens")

    with cf.ProcessPoolExecutor(max_workers=cfg.get("num_processes", C.DEFAULT_MAX_NUM_PROCESSES)) as executor:
        futures = []
        for idx, content in enumerate(io.read_until(cfg["input_file"], separator=[" "], bytes_mode=False)):
            futures.append(executor.submit(task, cfg, content, idx))

        for future in cf.as_completed(futures):
            future.result()

    reduce(output_folder)
    logger.info("Task Done!")


def task(config: dict, content: str, idx: int):
    tokenizer = tk.Tokenizer.from_file(**config["tokenizer"])
    token_ids = tokenizer.encode(content)
    np_token_ids = np.array(token_ids, dtype=np.int32)

    output_folder = config.get("output_folder", os.path.dirname(config["input_file"]))
    output_file = os.path.join(output_folder, f"{idx}.npy")
    np.save(output_file, np_token_ids)
    logger.info(f"Tokenized {len(content)} bytes into {np_token_ids.shape} tokens and saved to {output_file}")


if __name__ == "__main__":
    cli()
