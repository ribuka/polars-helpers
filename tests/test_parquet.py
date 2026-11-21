"""Tests for ``scan_parquet_relaxed``: inputs, order, types, and parts."""

import datetime as dt
import inspect
import os
import sys
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_helpers as plh
from polars_helpers import parquet
from polars_helpers._parquet_cache import Column


def naive(text: str) -> dt.datetime:
    """Parse a datetime without a time zone, as Parquet stores it.

    Parameters
    ----------
    text : str
        ISO 8601 datetime.

    Returns
    -------
    dt.datetime
        The naive datetime.
    """
    return dt.datetime.fromisoformat(text)


def write(path: Path, frame: pl.DataFrame) -> Path:
    """Write a frame to a Parquet file, creating its folder.

    Parameters
    ----------
    path : Path
        File to write.
    frame : pl.DataFrame
        Frame to write.

    Returns
    -------
    Path
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def scan(paths: object, root: Path, **kwargs: object) -> pl.LazyFrame:
    """Call ``scan_parquet_relaxed`` with parts kept in ``root``.

    Parameters
    ----------
    paths : object
        Paths to pass.
    root : Path
        Directory of the parts.
    **kwargs : object
        Other arguments.

    Returns
    -------
    pl.LazyFrame
        The result.
    """
    return plh.scan_parquet_relaxed(paths, directory=root, **kwargs)  # type: ignore[arg-type]


def part_files(root: Path) -> list[Path]:
    """List the parts kept in a root.

    Parameters
    ----------
    root : Path
        Directory of the parts.

    Returns
    -------
    list[Path]
        Every Parquet file under ``root``.
    """
    return sorted(root.rglob("*.parquet"))


def test_signature() -> None:
    """Take ``paths`` only by position and the options only by keyword."""
    params = inspect.signature(plh.scan_parquet_relaxed).parameters
    assert [(p.name, p.kind, p.default) for p in params.values()] == [
        ("paths", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("recursive", inspect.Parameter.KEYWORD_ONLY, False),
        ("columns", inspect.Parameter.KEYWORD_ONLY, None),
        ("directory", inspect.Parameter.KEYWORD_ONLY, None),
        ("cache_ttl_days", inspect.Parameter.KEYWORD_ONLY, 1),
    ]


def test_only_paths_returns_lazyframe(tmp_path: Path, isolated_cache: Path) -> None:
    """Return a plain LazyFrame with parts in the default root."""
    source = write(tmp_path / "a.parquet", pl.DataFrame({"x": [1, 2]}))
    out = plh.scan_parquet_relaxed(source)
    assert type(out) is pl.LazyFrame
    assert out.collect().to_dicts() == [{"x": 1}, {"x": 2}]
    assert part_files(isolated_cache)


# ---------------------------------------------------------------- inputs


def test_folder_is_natural_order(tmp_path: Path) -> None:
    """Stack the files of a folder with numbers in numeric order."""
    data = tmp_path / "data"
    for n in [10, 2, 1]:
        write(data / f"data_{n}.parquet", pl.DataFrame({"n": [n]}))
    out = scan(data, tmp_path / "cache").collect()
    assert out["n"].to_list() == [1, 2, 10]


def test_order_ignores_case_and_breaks_ties(tmp_path: Path) -> None:
    """Compare without case, then by the path itself."""
    data = tmp_path / "data"
    write(data / "B.parquet", pl.DataFrame({"n": ["B"]}))
    write(data / "a.parquet", pl.DataFrame({"n": ["a"]}))
    write(data / "f01.parquet", pl.DataFrame({"n": ["f01"]}))
    write(data / "f1.parquet", pl.DataFrame({"n": ["f1"]}))
    out = scan(data, tmp_path / "cache").collect()
    assert out["n"].to_list() == ["a", "B", "f01", "f1"]
    base = Path("/x").absolute()
    upper, lower = base / "A.parquet", base / "a.parquet"
    assert sorted([lower, upper], key=lambda p: parquet._sort_key(p, base)) == [upper, lower]


def test_order_by_component(tmp_path: Path) -> None:
    """Compare relative paths one component at a time."""
    data = tmp_path / "data"
    write(data / "s10" / "a.parquet", pl.DataFrame({"n": ["s10/a"]}))
    write(data / "s2" / "b.parquet", pl.DataFrame({"n": ["s2/b"]}))
    write(data / "s2.parquet", pl.DataFrame({"n": ["s2.parquet"]}))
    out = scan(data, tmp_path / "cache", recursive=True).collect()
    assert out["n"].to_list() == ["s2/b", "s2.parquet", "s10/a"]


def test_folder_skips_hidden_and_other_files(tmp_path: Path) -> None:
    """Read ``.parquet`` in any case, skipping ``.`` and ``_`` names."""
    data = tmp_path / "_root"
    write(data / "a.parquet", pl.DataFrame({"n": ["a"]}))
    write(data / "B.PARQUET", pl.DataFrame({"n": ["B"]}))
    write(data / ".hidden.parquet", pl.DataFrame({"n": ["hidden"]}))
    write(data / "_meta.parquet", pl.DataFrame({"n": ["meta"]}))
    write(data / "c.parquet.bak", pl.DataFrame({"n": ["bak"]}))
    write(data / "sub" / "d.parquet", pl.DataFrame({"n": ["sub/d"]}))
    write(data / "_sub" / "e.parquet", pl.DataFrame({"n": ["_sub/e"]}))
    write(data / ".sub" / "f.parquet", pl.DataFrame({"n": [".sub/f"]}))
    root = tmp_path / "cache"
    assert scan(data, root).collect()["n"].to_list() == ["a", "B"]
    assert scan(data, root, recursive=True).collect()["n"].to_list() == ["a", "B", "sub/d"]


def test_recursive_does_not_follow_folder_links(tmp_path: Path) -> None:
    """Skip symbolic links to folders."""
    data = tmp_path / "data"
    write(data / "a.parquet", pl.DataFrame({"n": ["a"]}))
    write(tmp_path / "other" / "b.parquet", pl.DataFrame({"n": ["b"]}))
    try:
        os.symlink(tmp_path / "other", data / "link", target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links cannot be created here")
    out = scan(data, tmp_path / "cache", recursive=True).collect()
    assert out["n"].to_list() == ["a"]


def test_explicit_files_are_not_filtered(tmp_path: Path) -> None:
    """Read explicit files whatever their names, in natural order."""
    files = [
        write(tmp_path / "d" / "x_10.bin", pl.DataFrame({"n": [10]})),
        write(tmp_path / "d" / ".x_2.parquet", pl.DataFrame({"n": [2]})),
        write(tmp_path / "d" / "_x_1.parquet", pl.DataFrame({"n": [1]})),
    ]
    out = scan([str(f) for f in files], tmp_path / "cache", recursive=True).collect()
    assert out["n"].to_list() == [2, 1, 10]


def test_files_order_from_common_parent(tmp_path: Path) -> None:
    """Order explicit files by their paths from their common parent."""
    b = write(tmp_path / "d" / "b10" / "x.parquet", pl.DataFrame({"n": ["b10"]}))
    a = write(tmp_path / "d" / "b9" / "x.parquet", pl.DataFrame({"n": ["b9"]}))
    out = scan([b, a], tmp_path / "cache").collect()
    assert out["n"].to_list() == ["b9", "b10"]


def test_duplicates_are_read_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read a file given twice, in any spelling, once."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"n": [1]}))
    monkeypatch.chdir(tmp_path)
    paths: list[str | Path] = [a, "a.parquet", str(tmp_path / "." / "a.parquet")]
    if sys.platform == "win32":
        paths.append(str(a).upper())
    assert scan(paths, tmp_path / "cache").collect()["n"].to_list() == [1]
    assert scan([tmp_path, str(tmp_path)], tmp_path / "cache").collect()["n"].to_list() == [1]


def test_single_item_sequence_is_a_single_path(tmp_path: Path) -> None:
    """Treat a sequence of one folder as that folder."""
    write(tmp_path / "d" / "a.parquet", pl.DataFrame({"n": [1]}))
    assert scan([tmp_path / "d"], tmp_path / "cache").collect()["n"].to_list() == [1]


@pytest.mark.parametrize(
    ("paths", "error", "match"),
    [
        ([], ValueError, "empty"),
        ("", ValueError, "empty"),
        ([""], ValueError, "empty"),
        (b"a.parquet", TypeError, "paths"),
        (123, TypeError, "paths"),
        ([123], TypeError, "paths"),
        ((p for p in ["a.parquet"]), TypeError, "paths"),
    ],
)
def test_invalid_paths(tmp_path: Path, paths: object, error: type[Exception], match: str) -> None:
    """Reject empty and wrongly typed paths."""
    with pytest.raises(error, match=match):
        scan(paths, tmp_path / "cache")


def test_missing_path(tmp_path: Path) -> None:
    """Raise FileNotFoundError naming the missing path."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"n": [1]}))
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        scan([a, tmp_path / "missing.parquet"], tmp_path / "cache")


def test_folder_must_be_alone(tmp_path: Path) -> None:
    """Reject a folder among other paths."""
    a = write(tmp_path / "d1" / "a.parquet", pl.DataFrame({"n": [1]}))
    write(tmp_path / "d2" / "b.parquet", pl.DataFrame({"n": [1]}))
    with pytest.raises(ValueError, match="folder"):
        scan([tmp_path / "d1", a], tmp_path / "cache")
    with pytest.raises(ValueError, match="folder"):
        scan([tmp_path / "d1", tmp_path / "d2"], tmp_path / "cache")


def test_folder_without_files(tmp_path: Path) -> None:
    """Reject a folder holding no Parquet file."""
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.csv").write_text("x\n1\n")
    with pytest.raises(ValueError, match="no Parquet file"):
        scan(tmp_path / "d", tmp_path / "cache")


def test_recursive_must_be_bool(tmp_path: Path) -> None:
    """Reject a ``recursive`` that is not a bool."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"n": [1]}))
    with pytest.raises(TypeError, match="recursive"):
        scan(a, tmp_path / "cache", recursive=1)


def test_invalid_input_writes_nothing(tmp_path: Path) -> None:
    """Check the inputs before creating the directory of the parts."""
    root = tmp_path / "cache"
    with pytest.raises(FileNotFoundError):
        scan(tmp_path / "missing.parquet", root)
    assert not root.exists()


# --------------------------------------------------------------- columns


def test_columns_none_keeps_every_column_in_first_order(tmp_path: Path) -> None:
    """Keep the union of the columns in the order they first appear."""
    a = write(tmp_path / "f1.parquet", pl.DataFrame({"b": [1], "a": ["x"]}))
    b = write(tmp_path / "f2.parquet", pl.DataFrame({"c": [True], "a": ["y"]}))
    out = scan([a, b], tmp_path / "cache").collect()
    assert out.columns == ["b", "a", "c"]
    assert out.to_dicts() == [
        {"b": 1, "a": "x", "c": None},
        {"b": None, "a": "y", "c": True},
    ]


def test_columns_selects_in_given_order(tmp_path: Path) -> None:
    """Keep only the given columns, in the given order, null where missing."""
    a = write(tmp_path / "f1.parquet", pl.DataFrame({"b": [1], "a": ["x"], "z": [1]}))
    b = write(tmp_path / "f2.parquet", pl.DataFrame({"a": ["y"], "z": ["s"]}))
    root = tmp_path / "cache"
    out = scan([a, b], root, columns=("a", "b")).collect()
    assert out.to_dicts() == [{"a": "x", "b": 1}, {"a": "y", "b": None}]
    assert len(part_files(root)) == 4
    names = {tuple(pl.read_parquet_schema(p)) for p in part_files(root)}
    assert names == {("a",), ("b",)}


def test_unselected_columns_need_no_common_type(tmp_path: Path) -> None:
    """Ignore type conflicts of columns that are not selected."""
    a = write(
        tmp_path / "f1.parquet",
        pl.DataFrame({"t": [naive("2020-01-01")], "x": [1]}).with_columns(
            pl.col("t").dt.replace_time_zone("UTC")
        ),
    )
    b = write(
        tmp_path / "f2.parquet",
        pl.DataFrame({"t": [naive("2020-01-01")], "x": [2]}).with_columns(
            pl.col("t").dt.replace_time_zone("Asia/Tokyo")
        ),
    )
    assert scan([a, b], tmp_path / "cache", columns=["x"]).collect()["x"].to_list() == [1, 2]


def test_unknown_column(tmp_path: Path) -> None:
    """Raise ColumnNotFoundError for a column in no file, before writing."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"x": [1]}))
    root = tmp_path / "cache"
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="nope"):
        scan(a, root, columns=["x", "nope"])
    assert not root.exists()


@pytest.mark.parametrize(
    ("columns", "error", "match"),
    [
        ([], ValueError, "empty"),
        (["x", "x"], ValueError, "more than once"),
        ("x", TypeError, "columns"),
        ((c for c in ["x"]), TypeError, "columns"),
        ([1], TypeError, "columns"),
        ({"x"}, TypeError, "columns"),
    ],
)
def test_invalid_columns(
    tmp_path: Path, columns: object, error: type[Exception], match: str
) -> None:
    """Reject empty, repeated, and wrongly typed columns before reading."""
    root = tmp_path / "cache"
    with pytest.raises(error, match=match):
        scan(tmp_path / "missing.parquet", root, columns=columns)
    assert not root.exists()


# ----------------------------------------------------------------- types


def tz(values: list[dt.datetime], zone: str) -> pl.Series:
    """Build a datetime series in a time zone.

    Parameters
    ----------
    values : list[dt.datetime]
        Naive datetimes.
    zone : str
        Time zone.

    Returns
    -------
    pl.Series
        The datetimes in ``zone``.
    """
    return pl.Series("t", values).dt.replace_time_zone(zone)


CASES: dict[str, list[pl.DataFrame]] = {
    "int32+float64": [
        pl.DataFrame({"c": [1, 2]}, schema={"c": pl.Int32}),
        pl.DataFrame({"c": [0.5]}),
    ],
    "string+int32": [
        pl.DataFrame({"c": ["a"]}),
        pl.DataFrame({"c": [7]}, schema={"c": pl.Int32}),
    ],
    "boolean+int32": [
        pl.DataFrame({"c": [True, False]}),
        pl.DataFrame({"c": [5]}, schema={"c": pl.Int32}),
    ],
    "date+datetime": [
        pl.DataFrame({"c": [dt.date(2020, 1, 2)]}),
        pl.DataFrame({"c": [naive("2021-03-04T05:06")]}),
    ],
    "datetime us+ms": [
        pl.DataFrame({"c": [naive("2020-01-01T00:00:00.123456")]}),
        pl.DataFrame({"c": [naive("2021-01-01")]}, schema={"c": pl.Datetime("ms")}),
    ],
    "list int32+int64": [
        pl.DataFrame({"c": [[1, 2]]}, schema={"c": pl.List(pl.Int32)}),
        pl.DataFrame({"c": [[3], []]}, schema={"c": pl.List(pl.Int64)}),
    ],
    "struct fields": [
        pl.DataFrame({"c": [{"a": 1}]}),
        pl.DataFrame({"c": [{"b": "x"}]}),
    ],
    "missing column and order": [
        pl.DataFrame({"a": [1], "b": ["x"]}),
        pl.DataFrame({"b": ["y"], "c": [True]}),
    ],
    "empty and non-empty": [
        pl.DataFrame({"a": []}, schema={"a": pl.Int8}),
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
    ],
    "all empty": [
        pl.DataFrame({"a": []}, schema={"a": pl.Int8}),
        pl.DataFrame({"b": []}, schema={"b": pl.String}),
    ],
    "three files": [
        pl.DataFrame({"a": [1], "b": ["x"]}, schema={"a": pl.Int8, "b": pl.String}),
        pl.DataFrame({"a": [300]}, schema={"a": pl.Int16}),
        pl.DataFrame({"c": [1.5], "a": [70000]}, schema={"c": pl.Float32, "a": pl.Int32}),
    ],
    "signed bounds": [
        pl.DataFrame({"c": [127, -128]}, schema={"c": pl.Int8}),
        pl.DataFrame({"c": [255, 0]}, schema={"c": pl.UInt8}),
    ],
    "64-bit bounds": [
        pl.DataFrame({"c": [2**63 - 1, -(2**63)]}, schema={"c": pl.Int64}),
        pl.DataFrame({"c": [2**64 - 1]}, schema={"c": pl.UInt64}),
    ],
    "float precision": [
        pl.DataFrame({"c": [0.1]}, schema={"c": pl.Float32}),
        pl.DataFrame({"c": [2**53 + 1]}, schema={"c": pl.Int64}),
    ],
    "null type": [
        pl.DataFrame({"c": [None, None]}, schema={"c": pl.Null}),
        pl.DataFrame({"c": [1, None]}, schema={"c": pl.Int16}),
    ],
    "all null type": [
        pl.DataFrame({"c": [None]}, schema={"c": pl.Null}),
        pl.DataFrame({"c": [None, None]}, schema={"c": pl.Null}),
    ],
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_diagonal_relaxed(tmp_path: Path, name: str) -> None:
    """Give the values, types, and order of ``diagonal_relaxed``."""
    frames = CASES[name]
    files = [write(tmp_path / f"f_{i}.parquet", f) for i, f in enumerate(frames)]
    expected = pl.concat(frames, how="diagonal_relaxed")
    for ttl in (None, 1):
        out = scan(files, tmp_path / f"cache-{ttl}", cache_ttl_days=ttl).collect()
        assert_frame_equal(out, expected)


def test_no_common_type(tmp_path: Path) -> None:
    """Raise ValueError naming the column, types, and files, before writing."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"t": tz([naive("2020-01-01")], "UTC")}))
    b = write(
        tmp_path / "b.parquet", pl.DataFrame({"t": tz([naive("2020-01-01")], "Asia/Tokyo")})
    )
    root = tmp_path / "cache"
    with pytest.raises(ValueError, match="'t'") as info:
        scan([a, b], root)
    message = str(info.value)
    assert "UTC" in message and "Asia/Tokyo" in message
    assert str(a) in message and str(b) in message
    assert isinstance(info.value.__cause__, pl.exceptions.PolarsError)
    assert not root.exists()


def test_no_column(tmp_path: Path) -> None:
    """Reject files that have no column at all."""
    a = write(tmp_path / "a.parquet", pl.DataFrame())
    with pytest.raises(ValueError, match="no column"):
        scan(a, tmp_path / "cache")


def test_empty_files_with_columns(tmp_path: Path) -> None:
    """Read files that have columns but no rows."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"x": []}, schema={"x": pl.Int64}))
    out = scan(a, tmp_path / "cache").collect()
    assert out.schema == pl.Schema({"x": pl.Int64})
    assert out.height == 0


# ----------------------------------------------------------------- parts


def test_one_part_per_file_and_column(tmp_path: Path) -> None:
    """Write a part per file and column, read by one scan per column."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"x": [1], "y": ["a"]}))
    b = write(tmp_path / "b.parquet", pl.DataFrame({"y": ["b"], "z": [1.5]}))
    root = tmp_path / "cache"
    out = scan([a, b], root)
    parts = part_files(root)
    assert len(parts) == 6
    assert all(len(pl.read_parquet_schema(p)) == 1 for p in parts)
    plan = out.explain()
    assert plan.startswith("HCONCAT")
    assert "UNION" not in plan
    assert plan.count("Parquet SCAN") == 3
    for line in plan.splitlines():
        if "Parquet SCAN" in line:
            assert line.count(".parquet") == 2


def test_one_column_is_one_scan(tmp_path: Path) -> None:
    """Return the scan of the parts itself for a single column."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"x": [1], "y": ["a"]}))
    plan = scan(a, tmp_path / "cache", columns=["y"]).explain()
    assert "HCONCAT" not in plan
    assert plan.count("Parquet SCAN") == 1


def test_parts_are_written_by_streaming_sinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write each part by a sink of one column, never by a collect."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}))
    b = write(tmp_path / "b.parquet", pl.DataFrame({"x": [3]}))
    sinks: list[list[str]] = []
    collects: list[list[str]] = []
    real_sink = pl.LazyFrame.sink_parquet
    real_collect = pl.LazyFrame.collect

    def sink(self: pl.LazyFrame, *args: object, **kwargs: object) -> object:
        sinks.append(self.collect_schema().names())
        return real_sink(self, *args, **kwargs)  # type: ignore[arg-type]

    def collect(self: pl.LazyFrame, *args: object, **kwargs: object) -> object:
        collects.append(self.collect_schema().names())
        return real_collect(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", sink)
    monkeypatch.setattr(pl.LazyFrame, "collect", collect)
    scan([a, b], tmp_path / "cache")
    assert sinks == [["x"], ["y"], ["x"], ["y"]]
    # Only the row count of ``b``, which lacks ``y``.
    assert collects == [["len"]]


def test_cast_error_keeps_type_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the polars error type, add a note, and delete unfinished parts."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"x": [1]}, schema={"x": pl.Int32}))
    b = write(tmp_path / "b.parquet", pl.DataFrame({"x": [2.5]}))
    root = tmp_path / "cache"
    real_writer = parquet._part_writer

    def failing_writer(path: Path, schema: pl.Schema) -> Callable[[Column, Path], None]:
        inner = real_writer(path, schema)

        def write_part(column: Column, out: Path) -> None:
            if path == b.resolve():
                inner(column, out)
                raise pl.exceptions.InvalidOperationError("injected")
            inner(column, out)

        return write_part

    monkeypatch.setattr(parquet, "_part_writer", failing_writer)
    for ttl in (None, 1):
        with pytest.raises(pl.exceptions.InvalidOperationError) as info:
            scan([a, b], root / str(ttl), cache_ttl_days=ttl)
        assert "injected" in str(info.value)
        assert not list((root / str(ttl)).rglob("*.tmp"))

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", _raise_compute)
    with pytest.raises(pl.exceptions.ComputeError) as info:
        scan([a, b], root / "note")
    notes = "\n".join(getattr(info.value, "__notes__", []))
    assert "'x'" in notes and str(a.resolve()) in notes
    assert "Int32" in notes and "Float64" in notes


def _raise_compute(*args: object, **kwargs: object) -> None:
    """Fail like a sink that cannot cast.

    Parameters
    ----------
    *args : object
        Ignored.
    **kwargs : object
        Ignored.

    Raises
    ------
    pl.exceptions.ComputeError
        Always.
    """
    raise pl.exceptions.ComputeError("cannot cast")


def test_change_while_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail with RuntimeError when a file changes while its parts are written."""
    a = write(tmp_path / "a.parquet", pl.DataFrame({"x": [1], "y": [2]}))
    real_writer = parquet._part_writer

    def changing_writer(path: Path, schema: pl.Schema) -> Callable[[Column, Path], None]:
        inner = real_writer(path, schema)

        def write_part(column: Column, out: Path) -> None:
            inner(column, out)
            write(a, pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))

        return write_part

    for ttl in (None, 1):
        root = tmp_path / f"cache-{ttl}"
        monkeypatch.setattr(parquet, "_part_writer", changing_writer)
        with pytest.raises(RuntimeError, match="changed"):
            scan(a, root, cache_ttl_days=ttl)
        assert not list(root.rglob("*.tmp"))
        assert not part_files(root)
        monkeypatch.setattr(parquet, "_part_writer", real_writer)
        assert scan(a, root, cache_ttl_days=ttl).collect()["x"].to_list() == [1, 2, 3]
