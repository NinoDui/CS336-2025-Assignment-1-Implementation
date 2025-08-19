import json
import pathlib
from typing import IO, BinaryIO, overload

import yaml

from cs336_basics.common import types as T


@overload
def load_config(path: T.PathLike) -> dict: ...


@overload
def load_config(path: BinaryIO) -> dict: ...


@overload
def load_config(path: IO[bytes]) -> dict: ...


def load_config(path: T.FileType) -> dict:
    if isinstance(path, BinaryIO | IO):
        try:
            file_name = path.name
        except AttributeError as err:
            raise ValueError(f"Can not get the file name from {path}") from err

        if file_name.endswith((".yaml", ".yml")):
            return yaml.safe_load(path)
        elif file_name.endswith(".json"):
            return json.load(path)
        else:
            raise ValueError(f"Unsupported file extension for {file_name}")

    path = pathlib.Path(path)

    with open(path, encoding="utf-8") as f:
        if path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif path.suffix in [".json"]:
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")
