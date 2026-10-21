"""Tests for natural sort keys."""

import pytest

from polars_helpers.utils import natural_keys


def test_natural_keys_orders_embedded_numbers_numerically() -> None:
    """Place ``item-2`` before ``item-10``."""
    values = ["item-10", "item-2", "item-1", "item-20"]

    assert sorted(values, key=natural_keys) == ["item-1", "item-2", "item-10", "item-20"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a12b3", ["a", 12, "b", 3, ""]),
        ("plain", ["plain"]),
        (7, [7]),
        (True, ["True"]),
    ],
)
def test_natural_keys_splits_text_and_digits(
    value: int | str | bool,
    expected: list[int | str],
) -> None:
    """Split text into text and integer fragments; keep ints and stringify bools."""
    assert natural_keys(value) == expected
