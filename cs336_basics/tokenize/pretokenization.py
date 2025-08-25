from collections.abc import Iterable
import logging
import os
import pathlib
from typing import BinaryIO

import click
import regex as re

from cs336_basics.common import constants as C, setup_logging, types as T

setup_logging()
logger = logging.getLogger(__name__)

__all__ = ["pretoken", "pretokenlize_in_parallel"]


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def _split(text: str, split_tokens: list[str] | None = None, reserve: bool = True) -> list[str]:
    """Split the text into segments, with certain split tokens.

    Args:
        text (str): The text to split
        split_tokens (list[str] | None, optional):
            The tokens to split the text. Defaults to None.
        reserve (bool, optional):
            Whether to reserve the split tokens. Defaults to True.

    Returns:
        list[str]: The segments of the text
    """
    if split_tokens is None:
        split_tokens = []
    if reserve:
        split_tokens = [re.escape(s) for s in split_tokens]
    split_pattern = f"(?:{'|'.join(split_tokens)})"
    return re.split(split_pattern, text, flags=re.UNICODE)


def _encode(token: str) -> T.BytesToken:
    # return tuple(x.encode("utf-8") for x in token)
    # treat multiple bytes as one token

    # ensure each bytes is converted to a single byte
    return tuple(bytes([b]) for b in token.encode("utf-8"))


def _pretoken_with_delimiters(
    text: str, special_tokens: list[str] | None = None, delimiters: str | Iterable[str] | None = None
) -> dict[str, int]:
    match delimiters:
        case str():
            delimiters = [delimiters]
        case Iterable():
            if not isinstance(delimiters, list):
                delimiters = list(delimiters)
        case None:
            delimiters = [" "]

    segments = _split(text, split_tokens=special_tokens, reserve=True)
    token_to_cnt: dict[str, int] = {}
    for segment in segments:
        for token in _split(segment, split_tokens=delimiters, reserve=False):
            token_to_cnt[token] = token_to_cnt.get(token, 0) + 1
    return token_to_cnt


def _pretoken_with_match_patten(
    text: str, special_tokens: list[str] | None = None, match_patten: str | None = None
) -> dict[str, int]:
    if match_patten is None:
        match_patten = C.PAT
    pattern = re.compile(match_patten)
    token_to_cnt: dict[str, int] = {}

    segments = _split(text, split_tokens=special_tokens, reserve=True)
    for segment in segments:
        for match in pattern.finditer(segment):
            token = match.group(0)
            token_to_cnt[token] = token_to_cnt.get(token, 0) + 1
    return token_to_cnt


def pretoken(
    chunk: str | bytes,
    special_tokens: list[str] | None = None,
    delimiters: str | Iterable[str] | None = None,
    match_patten: str | None = None,
) -> dict[T.BytesToken, int]:
    """Pretoken the trunk, with
        1) certain special tokens
        2) the split strategies

    Args:
        chunk (str | bytes):
            The chunk to pretokenize
        special_tokens (List[str] | None, optional):
            The special tokens to use.
            Defaults to None.
        delimiters (str | Iterable[str] | None, optional):
            The delimiters to use, like r' ''\t', <|endoftext|>
            Default to None
        match_patten (str | None, optional):
            The pattern to use for matching tokens.
            Defaults to PAT

    Returns:
        Dict[T.Token, int]: tokens and their frequencies
    """
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="ignore")

    if delimiters is not None:
        return {
            _encode(token): cnt
            for token, cnt in _pretoken_with_delimiters(
                chunk, special_tokens=special_tokens, delimiters=delimiters
            ).items()
        }
    elif match_patten is not None:
        return {
            _encode(token): cnt
            for token, cnt in _pretoken_with_match_patten(
                chunk, special_tokens=special_tokens, match_patten=match_patten
            ).items()
        }

    raise ValueError("No split strategy provided, either [delimiters] or [match_patten] must be provided.")


def pretokenlize_in_parallel(
    filepath: str | pathlib.Path,
    *,
    special_tokens: list[str] | None = None,
    delimiters: str | Iterable[str] | None = None,
    num_processes: int = 4,
) -> dict[T.BytesToken, int]:
    """Parallel wrapper for pretoken

    Args:
        filepath (str | pathlib.Path): Filepath to run pretoken
        num_processes (int, optional): Number of processes to use. Defaults to 4.

    Returns:
        dict[T.Token, int]: Token and their frequencies
    """
    raise NotImplementedError("Parallel pretokenization is not implemented yet")


@click.command()
@click.option("--file", "-f", required=True, help="Filepath to run pretoken")
@click.option("--parallel", "-p", is_flag=True, help="Enable parallel")
@click.option("--num_process", "-n", default=4, help="Number of Process")
def _run_pretoken(file: str, parallel: bool, num_process: int):
    if parallel:
        print("Pararllel")
    else:
        with open(file, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")
            for s, e in zip(boundaries[:-1], boundaries[1:], strict=False):
                f.seek(s)
                chunk = f.read(e - s).decode("utf-8", errors="ignore")  # bytes in, string out
                current_cnt = pretoken(chunk)
                print(current_cnt)


if __name__ == "__main__":
    _run_pretoken()
