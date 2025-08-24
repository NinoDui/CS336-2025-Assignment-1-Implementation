from collections.abc import Iterable

from cs336_basics.common import types as T


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_trategy: str | Iterable[str] | None = None,
) -> tuple[T.VocabType, T.MergeType]:
    return {}, {}  # type: ignore[return-value]
