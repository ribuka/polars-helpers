"""Tests for nested-column dropping and CSV output."""

from pathlib import Path

import polars as pl
import pytest

import polars_helpers as plh


@pytest.fixture
def nested_frame() -> pl.DataFrame:
    """Build a 2-row frame with flat and nested columns.

    Returns
    -------
    pl.DataFrame
        Flat columns ``id`` and ``text``, and nested columns ``lst``
        (List), ``arr`` (Array), and ``st`` (Struct).
    """
    return pl.DataFrame(
        {
            "id": [1, 2],
            "lst": [[1, 2], [3]],
            "text": ["a", "b"],
            "arr": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)),
            "st": [{"x": 1}, {"x": 2}],
        }
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_drop_nested_type_columns_drops_list_array_and_struct(
    nested_frame: pl.DataFrame,
    lazy: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drop every nested column, keep the frame kind, and print the names."""
    source = nested_frame.lazy() if lazy else nested_frame

    result = plh.drop_nested_type_columns(source)

    assert isinstance(result, type(source))
    collected = result.collect() if isinstance(result, pl.LazyFrame) else result
    assert collected.equals(nested_frame.select("id", "text"))
    assert "['lst', 'arr', 'st']" in capsys.readouterr().out


def test_safe_write_csv_writes_flat_columns(
    nested_frame: pl.DataFrame, tmp_path: Path
) -> None:
    """Write the flat columns of a frame that also has nested columns."""
    file_path = tmp_path / "out.csv"

    plh.safe_write_csv(nested_frame, file_path)

    assert pl.read_csv(file_path).equals(nested_frame.select("id", "text"))
