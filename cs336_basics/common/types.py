import os
import pathlib
from typing import IO, BinaryIO, TypeAlias

__all__ = ["TokenPair", "FileType"]

# Tokenize
TokenPair: TypeAlias = tuple[bytes, bytes]  # Example: {(a, b): 3}

# IO
PathLike: TypeAlias = str | os.PathLike | pathlib.Path
FileType: TypeAlias = PathLike | BinaryIO | IO[bytes]
