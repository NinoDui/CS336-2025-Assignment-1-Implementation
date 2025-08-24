import os
import pathlib
from typing import IO, BinaryIO, TypeAlias

__all__ = ["Token", "FileType"]

# Tokenize
Token: TypeAlias = tuple[bytes, ...]  # example ('l', 'o', 'w')
VocabType: TypeAlias = dict[int, bytes]
MergeType: TypeAlias = list[tuple[bytes, bytes]]

# IO
PathLike: TypeAlias = str | os.PathLike | pathlib.Path
FileType: TypeAlias = PathLike | BinaryIO | IO[bytes]
