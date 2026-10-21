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
    """Drop every nested column, keep the frame kind, and print nothing."""
    source = nested_frame.lazy() if lazy else nested_frame

    result = plh.drop_nested_type_columns(source)

    assert isinstance(result, type(source))
    collected = result.collect() if isinstance(result, pl.LazyFrame) else result
    assert collected.equals(nested_frame.select("id", "text"))
    assert capsys.readouterr().out == ""


def test_drop_nested_type_columns_prints_names_when_verbose(
    nested_frame: pl.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the dropped names only when ``verbose`` is set."""
    plh.drop_nested_type_columns(nested_frame, verbose=True)

    assert "['lst', 'arr', 'st']" in capsys.readouterr().out


@pytest.mark.parametrize("verbose", [False, True])
def test_safe_write_csv_writes_flat_columns(
    nested_frame: pl.DataFrame,
    tmp_path: Path,
    verbose: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write the flat columns, printing the result only when ``verbose``."""
    file_path = tmp_path / "out.csv"

    plh.safe_write_csv(nested_frame, file_path, verbose=verbose)

    assert pl.read_csv(file_path).equals(nested_frame.select("id", "text"))
    out = capsys.readouterr().out
    if verbose:
        assert "['lst', 'arr', 'st']" in out
        assert str(file_path) in out
    else:
        assert out == ""
