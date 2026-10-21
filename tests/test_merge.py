"""Tests for merging two frames with a merge-origin indicator."""

import polars as pl
import pytest

import polars_helpers as plh


@pytest.fixture
def left() -> pl.DataFrame:
    """Build the left frame keyed by ``k`` = 1, 2, 3.

    Returns
    -------
    pl.DataFrame
        Columns ``k`` and ``a``.
    """
    return pl.DataFrame({"k": [1, 2, 3], "a": ["x", "y", "z"]})


@pytest.fixture
def right() -> pl.DataFrame:
    """Build the right frame keyed by ``k`` = 2, 3, 4.

    Returns
    -------
    pl.DataFrame
        Columns ``k``, ``b``, and ``c``.
    """
    return pl.DataFrame({"k": [2, 3, 4], "b": [20, 30, 40], "c": ["p", "q", "r"]})


def _origin_counts(frame: pl.DataFrame, indicator: str = "_merge") -> dict[str, int]:
    """Count the rows of each merge origin.

    Parameters
    ----------
    frame : pl.DataFrame
        Merged frame.
    indicator : str, default "_merge"
        Name of the merge-origin column.

    Returns
    -------
    dict[str, int]
        Row count per merge origin.
    """
    return dict(frame.group_by(indicator).len().iter_rows())


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("left", {"both": 2, "left_only": 1}),
        ("right", {"both": 2, "right_only": 1}),
        ("full", {"both": 2, "left_only": 1, "right_only": 1}),
        ("inner", {"both": 2}),
    ],
)
def test_merge_marks_row_origin(
    left: pl.DataFrame,
    right: pl.DataFrame,
    how: plh.HOW,
    expected: dict[str, int],
) -> None:
    """Mark each row with where it came from for every join strategy."""
    result = plh.merge(left, right, on=["k"], how=how, show_result=False)

    assert isinstance(result, pl.DataFrame)
    assert _origin_counts(result) == expected


def test_merge_keeps_lazy_frames_lazy(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Return a LazyFrame matching the eager result for LazyFrame input."""
    eager = plh.merge(left, right, on=["k"], show_result=False)
    lazy = plh.merge(left.lazy(), right.lazy(), on=["k"], show_result=False)

    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(eager)


def test_merge_takes_only_selected_columns(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Take only ``columns`` from the right frame."""
    result = plh.merge(left, right, columns=["k", "b"], on=["k"], show_result=False)

    assert result.columns == ["k", "a", "b", "_merge"]


def test_merge_with_left_on_right_on_adds_right_key(left: pl.DataFrame) -> None:
    """Join on differently named keys, taking the right key even if not listed."""
    right = pl.DataFrame({"key": [1, 3], "b": [10, 30]})
    columns = ["b"]

    result = plh.merge(
        left, right, columns=columns, left_on="k", right_on="key", show_result=False,
    )

    assert result["b"].to_list() == [10, None, 30]
    assert _origin_counts(result) == {"both": 2, "left_only": 1}
    assert columns == ["b"]


def test_merge_names_or_omits_indicator(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Name the indicator column as asked, or leave it out for ``False``."""
    named = plh.merge(left, right, on=["k"], indicator="origin", show_result=False)
    omitted = plh.merge(left, right, on=["k"], indicator=False, show_result=False)

    assert "origin" in named.columns
    assert omitted.columns == ["k", "a", "b", "c"]


def test_merge_prints_shape_and_origin_counts(
    left: pl.DataFrame,
    right: pl.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the shape change and the merge-origin counts by default."""
    plh.merge(left, right, on=["k"])

    out = capsys.readouterr().out
    assert "(3, 2) -> (3, 5)" in out
    assert "left_only" in out


def test_merge_rejects_mixed_frame_kinds(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Reject a DataFrame merged with a LazyFrame."""
    with pytest.raises(TypeError, match="left: .* and right: .* must be the same type"):
        plh.merge(left, right.lazy(), on=["k"])


def test_merge_requires_keys(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Reject a merge without ``on`` or both ``left_on`` and ``right_on``."""
    with pytest.raises(ValueError, match="`left_on` and `right_on` must be specified"):
        plh.merge(left, right, left_on="k", show_result=False)
