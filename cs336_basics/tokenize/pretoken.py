from collections.abc import Iterable
import logging
import os
import pathlib
from typing import BinaryIO, overload

import click
import regex as re

from cs336_basics.common import constants as C, io, setup_logging, types as T

setup_logging()
logger = logging.getLogger(__name__)

__all__ = ["pretoken_and_count", "pretoken", "pretokenlize_in_parallel"]


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


def _split(
    text: str, split_tokens: Iterable[str] | None = None, *, reserve: bool = True, keep_split_tokens: bool = False
) -> list[str]:
    """Split the text into segments, with certain split tokens.

    Args:
        text (str): The text to split
        split_tokens (list[str] | None, optional):
            The tokens to split the text. Defaults to None.
        reserve (bool, optional):
            Whether to escape the characters in the split tokens
            Default to True

    Returns:
        list[str]: The segments of the text
    """
    if split_tokens is None:
        split_tokens = []
    if reserve:
        split_tokens = [re.escape(s) for s in split_tokens]

    # using capture group (without leading ?:) to include the split token in the result
    if keep_split_tokens:
        split_pattern = f"({'|'.join(split_tokens)})"
    else:
        split_pattern = f"(?:{'|'.join(split_tokens)})"
    return [x for x in re.split(split_pattern, text, flags=re.UNICODE) if x]


def pretoken_and_count(
    chunk: str | bytes,
    special_tokens: Iterable[str] | None = None,
    split_pattern: str | Iterable[str] | re.Pattern | None = None,
) -> dict[T.BytesToken, int]:
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="ignore")

    if split_pattern is None:
        split_pattern = re.compile(C.PAT)

    segments = _split(chunk, split_tokens=special_tokens, reserve=True, keep_split_tokens=False)

    token_to_cnt: dict[T.BytesToken, int] = {}
    for segment in segments:
        for token in _text_to_tokens(segment, split_pattern):
            token_to_cnt[token] = token_to_cnt.get(token, 0) + 1
    return token_to_cnt


def pretoken(
    chunk: str | bytes,
    special_tokens: Iterable[str] | None = None,
    split_pattern: str | Iterable[str] | re.Pattern | None = None,
) -> tuple[list[list[T.BytesToken]], dict[int, str]]:
    """Pretoken the text/bytes chunk to bytes tokens.

    Args:
        chunk (str | bytes): The text/bytes chunk
        special_tokens (list[str] | None, optional):
            The special tokens to split the text/bytes chunk. Defaults to None.
        pattern (str | re.Pattern | None, optional):
            The pattern to match the text/bytes chunk. Defaults to None.

    Returns:
        List[T.BytesToken]: The bytes tokens, splited by special tokens
    """
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="ignore")

    if special_tokens is None:
        return [_text_to_tokens(chunk, split_pattern)], {}
    elif not isinstance(special_tokens, set):
        # dict.fromkeys as a walkaround for ordered set, whose values are set to None by default # noqa: E501
        special_tokens = dict.fromkeys(sorted(special_tokens, key=len, reverse=True))

    segments: list[str] = _split(chunk, split_tokens=special_tokens, reserve=True, keep_split_tokens=True)

    pretokens: list[list[T.BytesToken]] = []
    sep_position: dict[int, str] = {}
    for idx, segment in enumerate(segments):
        if segment in special_tokens:
            # special token, unseparatable, e.g, [(b'special_token', )]
            pretokens.append([io.encode(segment, unseparable=True)])
            sep_position[idx] = segment
        else:
            pretokens.append(_text_to_tokens(segment, split_pattern))
    return pretokens, sep_position


@overload
def _text_to_tokens(text: str, pattern: str) -> list[T.BytesToken]: ...


@overload
def _text_to_tokens(text: str, pattern: Iterable[str]) -> list[T.BytesToken]: ...


@overload
def _text_to_tokens(text: str, pattern: re.Pattern) -> list[T.BytesToken]: ...


def _text_to_tokens(text: str, pattern: str | Iterable[str] | re.Pattern | None = None) -> list[T.BytesToken]:
    """Convert text string to bytes tokens (tuple of bytes),
    assuming no special tokens in the text

    Args:
        text (str): The input text
        pattern (str | Iterable[str] | re.Pattern | None, optional):
            The pattern to split the text. Defaults to PAT.

    Raises:
        ValueError: If the pattern is invalid

    Returns:
        list[T.BytesToken]:
            The bytes tokens in tuple for each elements separated by the pattern
    """
    if pattern is None:
        pattern = re.compile(C.PAT)

    match pattern:
        case str():
            return [io.encode(x) for x in _split(text, split_tokens=[pattern], reserve=False)]
        case Iterable():
            if not isinstance(pattern, list):
                pattern = list(pattern)
            return [io.encode(x) for x in _split(text, split_tokens=pattern, reserve=False)]
        case re.Pattern():
            return [io.encode(match.group(0)) for match in pattern.finditer(text)]
        case _:
            raise ValueError(f"Invalid pattern type: {pattern} of type {type(pattern)}")


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
