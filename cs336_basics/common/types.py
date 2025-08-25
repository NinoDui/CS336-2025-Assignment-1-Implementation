import os
import pathlib
from typing import IO, BinaryIO, TypeAlias

__all__ = ["BytesToken", "FileType"]

# Tokenize
BytesPair: TypeAlias = tuple[bytes, bytes]
BytesToken: TypeAlias = tuple[bytes, ...]  # example (b'l', b'o', b'w')
BytesTokenCount: TypeAlias = tuple[BytesToken, int]
VocabType: TypeAlias = dict[int, bytes]
MergeType: TypeAlias = list[tuple[bytes, bytes]]

# IO
PathLike: TypeAlias = str | os.PathLike | pathlib.Path
FileType: TypeAlias = PathLike | BinaryIO | IO[bytes]
