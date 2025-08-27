from collections.abc import Iterable
import os
import pathlib
import re
from typing import IO, Any, BinaryIO, TypeAlias, TypeVar

__all__ = ["BytesToken", "FileType"]

# Tokenize
BytesPair: TypeAlias = tuple[bytes, bytes]
BytesToken: TypeAlias = tuple[bytes, ...]  # example (b'l', b'o', b'w')
BytesTokenCount: TypeAlias = tuple[BytesToken, int]

VocabType: TypeAlias = dict[int, bytes]
MergeType: TypeAlias = list[tuple[bytes, bytes]]

ContentType = TypeVar("ContentType", str, bytes)
SplitTokenType: TypeAlias = (
    list[str]
    | set[str]
    | dict[str, Any]
    | tuple[str, ...]
    | list[bytes]
    | set[bytes]
    | dict[bytes, Any]
    | tuple[bytes, ...]
    | str
    | bytes
)
SplitPatternType: TypeAlias = str | Iterable[str] | re.Pattern | None

# IO
PathLike: TypeAlias = str | os.PathLike | pathlib.Path
FileType: TypeAlias = PathLike | BinaryIO | IO[bytes]
