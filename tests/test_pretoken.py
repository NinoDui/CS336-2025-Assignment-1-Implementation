from collections.abc import Callable
from functools import partial
from itertools import chain
import random
from typing import Any

import pytest

from cs336_basics.common import io, types as T
from cs336_basics.tokenize import pretoken as pre

WORD_ELEMENTS = ["Ashin", "Monster", "Ming", "Masa", "Stone"]
RESOURCES = {w: idx + 1 for idx, w in enumerate(WORD_ELEMENTS)}
DELIMITERS = [" ", "<|endoftext|>"]


def expected_counts():
    return {io.encode(x): cnt for x, cnt in RESOURCES.items()}


def text(need_shuffle: bool = False):
    all_words = []
    for w, cnt in RESOURCES.items():
        all_words += [w] * cnt

    if need_shuffle:
        random.shuffle(all_words)

    elems = []
    for w in all_words:
        if random.random() < 0.3:
            elems.append(random.choice(DELIMITERS[1:]))
        else:
            elems.append(DELIMITERS[0])
        elems.append(w)

    return "".join(elems)


def pretoken():
    result = []
    for w, cnt in RESOURCES.items():
        result += [io.encode(w)] * cnt
    return result


def _generate_test_case(
    num_cases: int, *, input_generator: Callable, output_generator: Callable
) -> list[tuple[Any, Any]]:
    return [(input_generator(), output_generator()) for _ in range(num_cases)]


@pytest.mark.parametrize(
    "test_case, expected_counts",
    _generate_test_case(5, input_generator=partial(text, need_shuffle=True), output_generator=expected_counts),
)
def test_pretoken_and_count(test_case: str, expected_counts: dict[str, int]):
    assert (
        pre.pretoken_and_count(test_case, special_tokens=[DELIMITERS[-1]], split_pattern=DELIMITERS[:-1])
        == expected_counts
    )


@pytest.mark.parametrize(
    "test_case, expected_output",
    _generate_test_case(5, input_generator=partial(text, need_shuffle=False), output_generator=pretoken),
)
def test_pretoken(test_case: str, expected_output: list[T.BytesToken]):
    tokens, sep_position = pre.pretoken(test_case, special_tokens=[DELIMITERS[-1]], split_pattern=DELIMITERS[:-1])

    assert list(chain.from_iterable(tokens)) == expected_output
    assert len(sep_position) is not None
