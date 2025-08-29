from collections.abc import Callable, Iterable
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


load_dict: Callable = load_config
load_vocab: Callable = load_config


@overload
def load_text(path: T.PathLike) -> str: ...


@overload
def load_text(path: BinaryIO) -> str: ...


@overload
def load_text(path: IO[bytes]) -> str: ...


def load_text(path: T.FileType) -> str:
    if isinstance(path, BinaryIO | IO):
        return path.read().decode("utf-8")
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()


load_sequence: Callable = load_text


def save_json(
    data: dict,
    path: T.PathLike,
):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


save_dict: Callable = save_json


@overload
def save_text(data: Iterable[str], path: T.PathLike, save_bytes: bool = False): ...


@overload
def save_text(data: Iterable[bytes], path: T.PathLike, save_bytes: bool = False): ...


def save_text(data: Iterable[str] | Iterable[bytes], path: T.PathLike, save_bytes: bool = False):
    if save_bytes:
        with open(path, "wb") as f:
            for line in data:
                line_bytes = line.encode("utf-8") if isinstance(line, str) else line
                f.write(line_bytes + b"\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            for line in data:
                line_str = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
                f.write(line_str + "\n")


save_sequence: Callable = save_text


def encode(token: str, *, unseparable: bool = False) -> T.BytesToken:
    # treat multiple bytes as one token
    # return tuple(x.encode("utf-8") for x in token)

    if unseparable:
        return (token.encode("utf-8"),)

    # ensure each bytes is converted to a single byte
    return tuple(bytes([b]) for b in token.encode("utf-8"))
