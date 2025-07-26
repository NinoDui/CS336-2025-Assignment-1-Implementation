import logging
import os
import pathlib
from typing import BinaryIO, Dict, List

import click

from cs336_basics.common import setup_logging
from cs336_basics.common.types import TokenPair

setup_logging()
logger = logging.getLogger(__name__)


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    logger.debug(
        f"Desire Chunks: {desired_num_chunks}, Special Tokens: {split_special_token!r}\nChunks: {chunk_boundaries}"  # noqa: E501
    )
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks   # noqa: E501
    return sorted(set(chunk_boundaries))


def pretoken(
    chunk: str | bytes, special_token: List[str] | None = None
) -> Dict[TokenPair, int]:
    if isinstance(chunk, str):
        chunk = chunk.encode("utf-8")

    pair_to_cnt: Dict = {}
    for pair in zip(chunk[:-1], chunk[1:]):
        pair_to_cnt[pair] = pair_to_cnt.get(pair, 0) + 1
    return pair_to_cnt


# Usage
def pretokenlize_in_parallel(
    filepath: str | pathlib.Path, num_processes: int = 4
) -> Dict[TokenPair, int] | None:
    return None


@click.command()
@click.option("--file", "-f", required=True, help="Filepath to run pretoken")
@click.option("--parallel", "-p", is_flag=True, help="Enable parallel")
@click.option("--num_process", "-n", default=4, help="Number of Process")
def run_pretoken(file: str, parallel: bool, num_process: int):
    if parallel:
        print("Pararllel")
    else:
        with open(file, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_process, "<|endoftext|>".encode("utf-8")
            )
            for s, e in zip(boundaries[:-1], boundaries[1:]):
                f.seek(s)
                chunk = f.read(e - s).decode(
                    "utf-8", errors="ignore"
                )  # bytes in, string out
                current_cnt = pretoken(chunk)
                print(current_cnt)


if __name__ == "__main__":
    run_pretoken()
