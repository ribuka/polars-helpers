"""Natural (human) sort key generation."""

from __future__ import annotations

import re

_DIGIT_RUNS = re.compile(r"(\d+)")


def _atoi(fragment: str) -> int | str:
    """Convert one split fragment to ``int`` when it is all digits.

    Parameters
    ----------
    fragment : str
        Fragment produced by splitting on digit runs.

    Returns
    -------
    int | str
        The fragment as an ``int`` if numeric, otherwise unchanged.
    """
    return int(fragment) if fragment.isdigit() else fragment


def natural_keys(text: int | str | bool) -> list[int | str]:
    """Build a sort key that orders embedded numbers numerically.

    Sorting by this key places ``"item-2"`` before ``"item-10"``, unlike plain
    lexicographic ordering. Booleans are treated as their text form so that
    they never compare as the integers ``0`` and ``1``.

    Parameters
    ----------
    text : int | str | bool
        Value to derive a sort key from.

    Returns
    -------
    list[int | str]
        Alternating text and integer fragments, comparable against keys
        built from other values of the same shape.
    """
    if isinstance(text, bool):
        text = str(text)

    if isinstance(text, int):
        return [text]

    return [_atoi(fragment) for fragment in _DIGIT_RUNS.split(str(text))]
