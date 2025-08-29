from collections.abc import Generator, Iterable
import concurrent.futures as cf
import logging
import multiprocessing as mp

import regex as re

from cs336_basics.common import constants as C, decorators as helper, io, types as T
from cs336_basics.tokenize import utils

logger = logging.getLogger(__name__)

__all__ = ["pretoken_and_count", "pretoken", "pretoken_and_count_in_parallel"]


@helper.timeit(log_level=logging.DEBUG)
def pretoken_and_count(
    chunk: str | bytes,
    special_tokens: Iterable[str] | None = None,
    split_pattern: str | Iterable[str] | re.Pattern | None = None,
) -> dict[T.BytesToken, int]:
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="ignore")

    if split_pattern is None:
        split_pattern = re.compile(C.PAT)

    segments = utils.split(chunk, split_tokens=special_tokens, reserve=True, keep_split_tokens=False)

    token_to_cnt: dict[T.BytesToken, int] = {}
    for segment in segments:
        for token in utils.text_to_tokens(segment, split_pattern):
            token_to_cnt[token] = token_to_cnt.get(token, 0) + 1

    logger.debug(f"[{mp.current_process().name} - Pretoken] Pretoken and update {len(token_to_cnt)} tokens")
    return token_to_cnt


@helper.timeit(log_level=logging.DEBUG)
def pretoken(
    chunk: str | bytes,
    *,
    special_tokens: Iterable[str] | None = None,
    split_pattern: str | Iterable[str] | re.Pattern | None = None,
) -> list[T.BytesToken]:
    """Pretoken the text/bytes chunk to bytes tokens.

    Args:
        chunk (str | bytes): The text/bytes chunk
        special_tokens (list[str] | None, optional):
            The special tokens to split the text/bytes chunk. Defaults to None.
        pattern (str | re.Pattern | None, optional):
            The pattern to match the text/bytes chunk. Defaults to None.

    Returns:
        list[T.BytesToken]: The bytes tokens, splited by special tokens
    """
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="ignore")

    if special_tokens is None:
        return utils.text_to_tokens(chunk, split_pattern)
    elif not isinstance(special_tokens, set):
        # dict.fromkeys as a walkaround for ordered set, whose values are set to None by default # noqa: E501
        special_tokens = dict.fromkeys(sorted(special_tokens, key=len, reverse=True))

    segments: list[str] = utils.split(chunk, split_tokens=special_tokens, reserve=True, keep_split_tokens=True)
    pretokens: list[T.BytesToken] = []
    for segment in segments:
        if segment in special_tokens:
            # special token, unseparatable, e.g, [(b'special_token', )]
            pretokens.append(io.encode(segment, unseparable=True))
        else:
            pretokens.extend(utils.text_to_tokens(segment, split_pattern))
    return pretokens


@helper.timeit()
def pretoken_and_count_in_parallel(
    filepath: str,
    *,
    special_tokens: list[str] | None = None,
    split_pattern: str | Iterable[str] | re.Pattern | None = None,
    num_processes: int = 4,
) -> dict[T.BytesToken, int]:
    """Parallel wrapper for pretoken

    Args:
        filepath (str | pathlib.Path): Filepath to run pretoken
        num_processes (int, optional): Number of processes to use. Defaults to 4.

    Returns:
        dict[T.Token, int]: Token and their frequencies
    """
    with cf.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures: list[cf.Future] = []
        for chunk in _read(filepath, split_special_token=b"<|endoftext|>"):
            futures.append(
                executor.submit(pretoken_and_count, chunk, special_tokens=special_tokens, split_pattern=split_pattern)
            )

        result: dict[T.BytesToken, int] = {}
        for future in futures:
            result = _reduce(result, future.result())

    return result


def _reduce(d1: dict[T.BytesToken, int], d2: dict[T.BytesToken, int]) -> dict[T.BytesToken, int]:
    for token, cnt in d2.items():
        d1[token] = d1.get(token, 0) + cnt
    return d1


def _read(file_path: str, split_special_token: str | bytes) -> Generator[bytes]:
    """Read the file and yield the chunks

    Args:
        file_path (str): The path to the file
        split_special_token (str | bytes): The special token to split the file
        num_consumers (int): The number of consumers

    Yields:
        bytes: The chunks
    """
    if isinstance(split_special_token, str):
        split_special_token = split_special_token.encode("utf-8")

    logger.info(f"[{mp.current_process().name} - Reader] Starting to read file {file_path}")

    with open(file_path, "rb") as f:
        buffer, chunk_idx, byte_cnt = b"", 0, 0
        while True:
            raw_content = f.read(C.DEFAULT_MAX_CHUNK_SIZE)
            if not raw_content or len(raw_content) == 0:
                break
            buffer += raw_content

            last_space_pos = buffer.rfind(b" ")
            last_split_special_token_pos = buffer.rfind(split_special_token)
            if last_space_pos != -1 and last_split_special_token_pos != -1:
                # both ' ' and split_special_token are found
                end_pos = min(last_space_pos, last_split_special_token_pos + len(split_special_token))
            elif last_space_pos != -1:
                # only ' ' is found
                end_pos = last_space_pos
            elif last_split_special_token_pos != -1:
                # only split_special_token is found
                end_pos = last_split_special_token_pos + len(split_special_token)
            else:
                # no ' ' or split_special_token is found
                end_pos = len(buffer)

            valid_content = buffer[:end_pos]
            buffer = buffer[end_pos:] if end_pos < len(buffer) else b""

            chunks = utils.split(valid_content, split_tokens=[split_special_token], keep_split_tokens=False)
            for chunk in chunks:
                yield chunk

                # monitor status
                chunk_idx += 1
                byte_cnt += len(chunk)

                if chunk_idx % 1_000_000 == 0:
                    logger.info(
                        f"[{mp.current_process().name} - Reader] Read {chunk_idx + 1}th chunks, {len(chunk)} bytes"
                    )

    logger.info(f"[{mp.current_process().name} - Reader] Finished reading {byte_cnt} bytes from file {file_path}")
