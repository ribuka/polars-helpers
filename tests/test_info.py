"""Tests for the frame information helpers."""

import decimal
from collections.abc import Callable

import polars as pl
import pytest

import polars_helpers as plh


@pytest.fixture
def mixed_dtype_frame() -> pl.DataFrame:
    """Build a 3-row frame with numeric and non-numeric dtypes.

    Returns
    -------
    pl.DataFrame
        Numeric columns ``i8``, ``u32``, ``i64``, ``f32``, and ``dec``, and
        non-numeric columns ``text`` and ``flag``.
    """
    return pl.DataFrame(
        {
            "i8": pl.Series([1, 2, 3], dtype=pl.Int8),
            "u32": pl.Series([1, 2, 3], dtype=pl.UInt32),
            "text": ["a", "b", "c"],
            "i64": [1, 2, 3],
            "f32": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
            "flag": [True, False, True],
            "dec": pl.Series(
                [decimal.Decimal("1.5")] * 3, dtype=pl.Decimal(10, 2)
            ),
        }
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_get_shape_counts_rows_and_columns(
    mixed_dtype_frame: pl.DataFrame, lazy: bool
) -> None:
    """Return ``(rows, columns)`` for both frame kinds."""
    source = mixed_dtype_frame.lazy() if lazy else mixed_dtype_frame

    assert plh.get_shape(source) == (3, 7)


@pytest.mark.parametrize("lazy", [False, True])
def test_get_columns_and_schema_keep_column_order(
    mixed_dtype_frame: pl.DataFrame, lazy: bool
) -> None:
    """Return names and dtypes in column order, optionally restricted."""
    source = mixed_dtype_frame.lazy() if lazy else mixed_dtype_frame

    assert plh.get_columns(source) == mixed_dtype_frame.columns
    assert dict(plh.get_schema(source)) == dict(mixed_dtype_frame.schema)
    assert dict(plh.get_schema(source, ["text", "i8"])) == {
        "text": pl.String,
        "i8": pl.Int8,
    }


@pytest.mark.parametrize("helper", [plh.get_shape, plh.get_columns, plh.get_schema])
def test_info_helpers_reject_non_frames(helper: Callable[..., object]) -> None:
    """Raise ``TypeError`` for inputs that are not polars frames."""
    with pytest.raises(TypeError, match="must be a polars"):
        helper({"a": [1]})


@pytest.mark.parametrize(
    "helper",
    [
        lambda df: plh.get_value_counts(df, "a"),
        lambda df: plh.get_schema(df, ["a"]),
    ],
)
def test_column_helpers_reject_non_frames(helper: Callable[..., object]) -> None:
    """Raise ``TypeError`` before selecting columns from a non-frame input."""
    with pytest.raises(TypeError, match="must be a polars"):
        helper({"a": [1]})


def test_get_numeric_args_includes_every_numeric_dtype(
    mixed_dtype_frame: pl.DataFrame,
) -> None:
    """Pick signed, unsigned, float, and decimal columns, in column order."""
    assert plh.get_numeric_args(mixed_dtype_frame) == [
        "i8",
        "u32",
        "i64",
        "f32",
        "dec",
    ]


def test_get_numeric_args_includes_int128() -> None:
    """Pick ``Int128`` columns where polars supports them."""
    if not hasattr(pl, "Int128"):
        pytest.skip("this polars has no Int128")
    frame = pl.DataFrame({"big": pl.Series([1], dtype=pl.Int128), "s": ["a"]})

    assert plh.get_numeric_args(frame) == ["big"]


def test_get_numeric_args_limits_to_args(
    mixed_dtype_frame: pl.DataFrame,
) -> None:
    """Consider only the given candidate columns."""
    assert plh.get_numeric_args(mixed_dtype_frame, ["text", "u32", "flag"]) == [
        "u32"
    ]


@pytest.mark.parametrize("lazy", [False, True])
def test_get_unique_values_sorts_numbers_numerically(lazy: bool) -> None:
    """Sort numeric values by value, not by their string form."""
    frame = pl.DataFrame({"f": [1.5, 1.25, None, -2.0, 10.0, 1.5]})
    source = frame.lazy() if lazy else frame

    assert plh.get_unique_values(source, "f") == [-2.0, 1.25, 1.5, 10.0]


@pytest.mark.parametrize("lazy", [False, True])
def test_get_unique_values_sorts_strings_naturally(lazy: bool) -> None:
    """Sort strings with embedded numbers numerically, dropping nulls."""
    frame = pl.DataFrame({"s": ["item-10", "item-2", None, "item-1", "item-2"]})
    source = frame.lazy() if lazy else frame

    assert plh.get_unique_values(source, "s") == ["item-1", "item-2", "item-10"]


def test_get_unique_values_without_sort_returns_distinct_values() -> None:
    """Return each distinct non-null value once when not sorting."""
    frame = pl.DataFrame({"i": [3, 1, None, 3, 2]})

    result = plh.get_unique_values(frame, "i", sort=False)

    assert sorted(result) == [1, 2, 3]


def test_get_unique_values_rejects_non_frames() -> None:
    """Raise ``TypeError`` for inputs that are not polars frames."""
    with pytest.raises(TypeError, match="must be a polars"):
        plh.get_unique_values({"a": [1]}, "a")  # ty:ignore[invalid-argument-type]


@pytest.mark.parametrize("lazy", [False, True])
def test_get_value_counts_orders_by_count_then_value(lazy: bool) -> None:
    """Sort by descending count and break ties by ascending value."""
    frame = pl.DataFrame({"k": ["d", "a", "c", "c", "a", "c", "b"]})
    source = frame.lazy() if lazy else frame

    result = plh.get_value_counts(source, "k")

    assert result.rows() == [("c", 3), ("a", 2), ("b", 1), ("d", 1)]


@pytest.mark.parametrize("lazy", [False, True])
def test_column_helpers_read_wildcard_column_name_literally(lazy: bool) -> None:
    """Read a column named ``*`` as itself, not as a wildcard."""
    frame = pl.DataFrame({"other": [1, 1, 2], "*": ["x2", "x10", None]})
    source = frame.lazy() if lazy else frame

    assert plh.get_value_counts(source, "*").rows() == [
        (None, 1),
        ("x10", 1),
        ("x2", 1),
    ]
    assert plh.get_unique_values(source, "*") == ["x2", "x10"]
