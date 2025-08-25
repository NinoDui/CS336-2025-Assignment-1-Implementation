import random

import pytest

from cs336_basics.tokenize import pretokenization

WORD_ELEMENTS = ["Ashin", "Monster", "Ming", "Masa", "Stone"]
RESOURCES = {w: idx + 1 for idx, w in enumerate(WORD_ELEMENTS)}
DELIMITERS = [" ", "<|endoftext|>"]


def _generate_test_case(num_cases: int, *, delimiters: list[str]) -> list[tuple[str, dict[tuple[bytes, ...], int]]]:
    test_cases, all_words = [], []
    expected_counts = {tuple(x.encode("utf-8") for x in w): cnt for w, cnt in RESOURCES.items()}

    for w, cnt in RESOURCES.items():
        all_words += [w] * cnt

    for _ in range(num_cases):
        random.shuffle(all_words)
        elems = [all_words[0]]
        for w in all_words[1:]:
            if random.random() < 0.3:
                elems.append(random.choice(delimiters[1:]))
            else:
                elems.append(delimiters[0])
            elems.append(w)
        test_cases.append(("".join(elems), expected_counts))

    return test_cases


@pytest.mark.parametrize("test_case, expected_counts", _generate_test_case(5, delimiters=DELIMITERS))
def test_pretoken(test_case: str, expected_counts: dict[str, int]):
    assert (
        pretokenization.pretoken(test_case, special_tokens=[DELIMITERS[-1]], delimiters=DELIMITERS[:-1])
        == expected_counts
    )
