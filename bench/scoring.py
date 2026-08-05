"""Answer normalisation and scoring.

Scoring compares strings, so the normaliser decides how much of the score is
about the model and how much is about formatting. It strips the things a correct
answer may differ by — thousands separators, currency words, case, trailing
punctuation — and nothing else. It does **not** try to extract a number from a
sentence, because a rule that liberal starts marking wrong answers correct
whenever the right number happens to appear somewhere in the reasoning.
"""

from __future__ import annotations

import re

CURRENCY_WORDS = (
    "euros",
    "euro",
    "eur",
    "€",
)

_WHITESPACE = re.compile(r"\s+")
_TRAILING_ZEROS = re.compile(r"^(-?\d+)\.0+$")


def normalise(text: str) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = value.replace(",", "")
    for word in CURRENCY_WORDS:
        value = value.replace(word, " ")
    value = value.replace("$", " ")
    value = _WHITESPACE.sub(" ", value).strip()
    value = value.strip(" .;:!")
    # "9450.00" and "9450" are the same answer.
    collapsed = _TRAILING_ZEROS.match(value)
    if collapsed:
        value = collapsed.group(1)
    return value


def score(answer: str | None, expected: str, match: str = "contains") -> bool:
    if answer is None:
        return False
    got = normalise(answer)
    want = normalise(expected)
    if not want:
        return False
    if match == "exact":
        return got == want
    return want in got


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]
