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


def _origin_by_key(frame: pl.DataFrame, key: str = "k") -> dict[int, str]:
    """Map each key to its merge origin.

    Parameters
    ----------
    frame : pl.DataFrame
        Merged frame with the ``_merge`` column.
    key : str, default "k"
        Name of the key column.

    Returns
    -------
    dict[int, str]
        Merge origin per key.
    """
    return dict(frame.select(key, "_merge").iter_rows())


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("left", {1: "left_only", 2: "both", 3: "both"}),
        ("right", {2: "both", 3: "both", 4: "right_only"}),
        ("full", {1: "left_only", 2: "both", 3: "both", 4: "right_only"}),
        ("inner", {2: "both", 3: "both"}),
    ],
)
def test_merge_marks_row_origin_with_one_key_column(
    left: pl.DataFrame,
    right: pl.DataFrame,
    how: plh.HOW,
    expected: dict[int, str],
) -> None:
    """Mark each row's origin and coalesce the keys for every join strategy."""
    result = plh.merge(left, right, on="k", how=how)

    assert isinstance(result, pl.DataFrame)
    assert sorted(result.columns) == ["_merge", "a", "b", "c", "k"]
    assert _origin_by_key(result) == expected


def test_merge_keeps_lazy_frames_lazy(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Return a LazyFrame matching the eager result for LazyFrame input."""
    eager = plh.merge(left, right, on=["k"])
    lazy = plh.merge(left.lazy(), right.lazy(), on=["k"])

    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(eager)


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("left", [("a", 30), ("b", 10), ("b", 11), ("c", 20), ("d", 10), ("d", 11), ("e", None)]),
        ("inner", [("a", 30), ("b", 10), ("b", 11), ("c", 20), ("d", 10), ("d", 11)]),
        (
            "right",
            [(None, 40), ("b", 10), ("d", 10), ("c", 20), ("b", 11), ("d", 11), (None, 60), ("a", 30)],
        ),
        (
            "full",
            [
                ("a", 30), ("b", 10), ("b", 11), ("c", 20), ("d", 10), ("d", 11), ("e", None),
                (None, 40), (None, 60),
            ],
        ),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("indicator", ["_merge", None])
def test_merge_keeps_row_order_like_pandas(
    how: plh.HOW,
    expected: list[tuple[str | None, int | None]],
    lazy: bool,
    indicator: str | None,
) -> None:
    """Keep the row order of ``pandas.merge`` for eager and lazy frames."""
    left = pl.DataFrame({"k": [3, 1, 2, 1, 5], "a": ["a", "b", "c", "d", "e"]})
    right = pl.DataFrame({"k": [4, 1, 2, 1, 6, 3], "b": [40, 10, 20, 11, 60, 30]})

    if lazy:
        result = plh.merge(left.lazy(), right.lazy(), on="k", how=how, indicator=indicator)
        result = result.collect()
    else:
        result = plh.merge(left, right, on="k", how=how, indicator=indicator)

    assert list(result.select("a", "b").iter_rows()) == expected


@pytest.mark.parametrize("columns", [["k", "b"], ["b"]])
def test_merge_takes_selected_columns_and_the_keys(
    left: pl.DataFrame,
    right: pl.DataFrame,
    columns: list[str],
) -> None:
    """Take only ``columns`` from the right frame, adding its keys if left out."""
    result = plh.merge(left, right, columns=columns, on=["k"])

    assert result.columns == ["k", "a", "b", "_merge"]


def test_merge_with_left_on_right_on_coalesces_into_left_key(left: pl.DataFrame) -> None:
    """Join on differently named keys, keeping the left key name only."""
    right = pl.DataFrame({"key": [1, 3, 5], "b": [10, 30, 50]})
    columns = ["b"]

    result = plh.merge(left, right, columns=columns, left_on="k", right_on="key", how="full")

    assert result.columns == ["k", "a", "b", "_merge"]
    assert _origin_by_key(result) == {1: "both", 2: "left_only", 3: "both", 5: "right_only"}
    assert columns == ["b"]


def test_merge_accepts_several_keys() -> None:
    """Join on several key columns given as lists."""
    left = pl.DataFrame({"x": [1, 1, 2], "y": ["a", "b", "a"], "v": [1, 2, 3]})
    right = pl.DataFrame({"x2": [1, 2], "y2": ["b", "a"], "w": [10, 20]})

    result = plh.merge(left, right, left_on=["x", "y"], right_on=["x2", "y2"])

    assert result["w"].to_list() == [None, 10, 20]
    assert result["_merge"].to_list() == ["left_only", "both", "both"]


def test_merge_names_or_omits_indicator(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Name the indicator column as asked, or leave it out for ``None``."""
    named = plh.merge(left, right, on=["k"], indicator="origin")
    omitted = plh.merge(left, right, on=["k"], indicator=None)

    assert named.columns == ["k", "a", "b", "c", "origin"]
    assert omitted.columns == ["k", "a", "b", "c"]


def test_merge_keeps_columns_named_like_its_temporary_flags(right: pl.DataFrame) -> None:
    """Keep user columns that a naive implementation would use as flags."""
    left = pl.DataFrame({"k": [1, 2], "in_left": ["p", "q"], "in_right": ["r", "s"]})

    result = plh.merge(left, right, on=["k"])

    assert result.columns == ["k", "in_left", "in_right", "b", "c", "_merge"]


def test_merge_rejects_indicator_that_names_an_existing_column(
    left: pl.DataFrame,
    right: pl.DataFrame,
) -> None:
    """Refuse to overwrite a column with the indicator."""
    with pytest.raises(ValueError, match="indicator column 'b' already exists"):
        plh.merge(left, right, on=["k"], indicator="b")


def test_merge_is_silent_by_default(
    left: pl.DataFrame,
    right: pl.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print nothing unless ``verbose`` is set."""
    plh.merge(left, right, on=["k"])

    assert capsys.readouterr().out == ""


def test_merge_verbose_prints_shape_and_origin_counts(
    left: pl.DataFrame,
    right: pl.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the shape change and the merge-origin counts with ``verbose``."""
    plh.merge(left, right, on=["k"], verbose=True)

    out = capsys.readouterr().out
    assert "(3, 2) -> (3, 5)" in out
    assert out.index("both") < out.index("left_only")


@pytest.mark.parametrize("indicator", ["_merge", None])
def test_merge_verbose_runs_a_lazy_join_once(
    left: pl.DataFrame,
    right: pl.DataFrame,
    indicator: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Evaluate the lazy join only once to print the verbose summary."""
    calls = []

    def count_calls(s: pl.Series) -> pl.Series:
        calls.append(len(s))
        return s

    counted = right.lazy().with_columns(
        pl.col("k").map_batches(count_calls, return_dtype=pl.Int64)
    )

    plh.merge(left.lazy(), counted, on=["k"], indicator=indicator, verbose=True)

    assert len(calls) == 1
    assert "(3, 2) -> (3, " in capsys.readouterr().out


def test_merge_without_indicator_adds_no_temporary_flags(
    left: pl.DataFrame, right: pl.DataFrame
) -> None:
    """Leave the temporary origin flags out of the query without an indicator."""
    plan = plh.merge(left.lazy(), right.lazy(), on=["k"], indicator=None).explain(optimized=False)

    assert "__polars_helpers_in" not in plan


def test_merge_rejects_mixed_frame_kinds(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Reject a DataFrame merged with a LazyFrame."""
    with pytest.raises(TypeError, match="left: .* and right: .* must be the same type"):
        plh.merge(left, right.lazy(), on=["k"])


def test_merge_rejects_two_non_polars_values() -> None:
    """Reject two values of the same type that are not polars frames."""
    with pytest.raises(TypeError, match="must be the same type"):
        plh.merge({"k": [1]}, {"k": [1]}, on=["k"])  # ty:ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        ({"left_on": "k"}, "`left_on` and `right_on` must be specified"),
        ({}, "`left_on` and `right_on` must be specified"),
        ({"on": "k", "left_on": "k", "right_on": "k"}, "not both"),
    ],
)
def test_merge_requires_one_way_of_giving_keys(
    left: pl.DataFrame,
    right: pl.DataFrame,
    keys: dict[str, str],
    message: str,
) -> None:
    """Accept either ``on`` or both ``left_on`` and ``right_on``."""
    with pytest.raises(ValueError, match=message):
        plh.merge(left, right, **keys)  # ty:ignore[invalid-argument-type]
