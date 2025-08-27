from collections.abc import Iterable
import logging
import os
from typing import BinaryIO, overload

import regex as re

from cs336_basics.common import constants as C, io, types as T

logger = logging.getLogger(__name__)


@overload
def split(
    text: str,
    *,
    split_tokens: Iterable[str] | None = None,
    reserve: bool = True,
    keep_split_tokens: bool = False,
) -> list[str]: ...


@overload
def split(
    text: bytes,
    *,
    split_tokens: Iterable[bytes] | None = None,
    reserve: bool = True,
    keep_split_tokens: bool = False,
) -> list[bytes]: ...


def split(
    text: str | bytes,
    *,
    split_tokens: Iterable[str | bytes] | None = None,
    reserve: bool = True,
    keep_split_tokens: bool = False,
):
    """Split the text into segments, with certain split tokens.

    Args:
        text (str | bytes): The text to split
        split_tokens (SplitTokenType[str] | SplitTokenType[bytes] | None, optional):
            The tokens to split the text. Defaults to None.
        reserve (bool, optional):
            Whether to escape the characters in the split tokens
            Default to True
        keep_split_tokens (bool, optional):
            Whether to keep the split tokens in the result
            Default to False

    Raises:
        ValueError: If the text is not str or bytes

    Returns:
        list[str | bytes]: The segments of the text
    """
    match split_tokens:
        case None:
            split_tokens = []
        case dict() | set() | tuple():
            split_tokens = list(split_tokens)
        case str() | bytes():
            split_tokens = [split_tokens]
        case _:
            if not isinstance(split_tokens, list):
                raise ValueError(f"Invalid split tokens type: {type(split_tokens)}")

    splitters: list[str | bytes] = []
    if isinstance(text, str):
        for token in split_tokens:
            match token:
                case str():
                    splitters.append(token)
                case bytes():
                    splitters.append(token.decode("utf-8", errors="replace"))
                case _:
                    raise ValueError(f"Invalid split token type: {type(token)} in {split_tokens}")
        if reserve:
            splitters = [re.escape(s) for s in splitters]
        return _split_str(text, splitters, keep_split_tokens=keep_split_tokens)  # type: ignore[arg-type]
    elif isinstance(text, bytes):
        for token in split_tokens:
            match token:
                case bytes():
                    splitters.append(token)
                case str():
                    splitters.append(token.encode("utf-8"))
                case _:
                    raise ValueError(f"Invalid split token type: {type(token)} in {split_tokens}")
        if reserve:
            splitters = [re.escape(s) for s in splitters]
        return _split_bytes(text, splitters, keep_split_tokens=keep_split_tokens)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Invalid text type: {type(text)}")


def _split_str(text: str, splitter: Iterable[str], *, keep_split_tokens: bool = False) -> list[str]:
    if keep_split_tokens:
        split_pattern = f"({'|'.join(splitter)})"
    else:
        split_pattern = f"(?:{'|'.join(splitter)})"
    return [x for x in re.split(split_pattern, text, flags=re.UNICODE) if x]


def _split_bytes(text: bytes, splitter: Iterable[bytes], *, keep_split_tokens: bool = False) -> list[bytes]:
    if keep_split_tokens:
        split_pattern = b"(" + b"|".join(splitter) + b")"
    else:
        split_pattern = b"|".join(splitter)
    return [x for x in re.split(split_pattern, text) if x]


@overload
def text_to_tokens(text: str, pattern: str) -> list[T.BytesToken]: ...


@overload
def text_to_tokens(text: str, pattern: Iterable[str]) -> list[T.BytesToken]: ...


@overload
def text_to_tokens(text: str, pattern: re.Pattern) -> list[T.BytesToken]: ...


def text_to_tokens(text: str, pattern: str | Iterable[str] | re.Pattern | None = None) -> list[T.BytesToken]:
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
            return [io.encode(x) for x in split(text, split_tokens=[pattern], reserve=False)]
        case dict() | set() | tuple() | list():
            if not isinstance(pattern, list):
                pattern = list(pattern)
            return [io.encode(x) for x in split(text, split_tokens=pattern, reserve=False)]
        case re.Pattern():
            return [io.encode(match.group(0)) for match in pattern.finditer(text)]
        case _:
            raise ValueError(f"Invalid pattern type: {pattern} of type {type(pattern)}")


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
