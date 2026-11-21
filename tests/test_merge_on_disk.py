"""Tests for merging two LazyFrames on disk with ``merge_on_disk``."""

from __future__ import annotations

import importlib
import itertools
import json
import os
import random
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import helpers
import polars as pl
import pytest

import polars_helpers as plh
from polars_helpers import _disk

merge_on_disk_module = importlib.import_module("polars_helpers.merge_on_disk")

# Name of a result file of ``merge_on_disk``.
_RESULT = re.compile(r"merge-[0-9a-f]{32}-\d{5}\.parquet")

HOWS: list[plh.HOW] = ["left", "inner", "right", "full"]


# ``_restore_limit`` before ``small_restore`` replaces it.
_restore_limit = merge_on_disk_module._restore_limit


@pytest.fixture(autouse=True)
def small_restore(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    """Put the merged rows back in order a few at a time.

    The rows are counted by intervals of ten row numbers. Parametrize it
    indirectly with the most merged rows of one range; the default ``3``
    makes almost every interval be counted again row number by row number,
    while ``50`` groups whole intervals into one range.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Request whose ``param``, if any, is the most merged rows of one range.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace ``_restore_limit`` and ``_COUNT_BLOCK``.

    Returns
    -------
    int
        Most merged rows of one range.
    """
    limit: int = getattr(request, "param", 3)
    monkeypatch.setattr(merge_on_disk_module, "_restore_limit", lambda merged: limit)
    monkeypatch.setattr(merge_on_disk_module, "_COUNT_BLOCK", 10)
    return limit


def _key_values(kind: str, ids: list[int | None]) -> dict[str, list[Any]]:
    """Build the key columns of one side from integer ids.

    Parameters
    ----------
    kind : str
        ``"int"``, ``"str"``, ``"int_str"``, ``"left_named"``, or
        ``"right_named"``.
    ids : list[int | None]
        Id of each row; ``None`` gives null keys.

    Returns
    -------
    dict[str, list[Any]]
        Key columns by name.
    """
    as_str = [None if i is None else f"LOT-{i:04d}" for i in ids]
    if kind == "int":
        return {"k": ids}
    if kind == "str":
        return {"k": as_str}
    if kind == "int_str":
        return {"k1": [None if i is None else i % 7 for i in ids], "k2": as_str}
    return {"lk" if kind == "left_named" else "rk": ids}


def _frames(
    kind: str, right_unique: bool, nulls: bool, seed: int
) -> tuple[pl.LazyFrame, pl.LazyFrame, dict[str, Any]]:
    """Build small random frames to merge and the key arguments.

    Parameters
    ----------
    kind : str
        ``"int"``, ``"str"``, ``"int_str"``, or ``"named"``.
    right_unique : bool
        Whether the keys of the right frame are unique.
    nulls : bool
        Whether some keys of both sides are null.
    seed : int
        Seed of the random data.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame, dict[str, Any]]
        Left frame of 200 rows, right frame of 40 rows, and the key
        arguments to pass to the merge.
    """
    rng = random.Random(seed)
    domain = 50
    left_ids: list[int | None] = [rng.randrange(domain) for _ in range(200)]
    right_ids: list[int | None] = (
        list(rng.sample(range(domain), 40)) if right_unique
        else [rng.randrange(domain) for _ in range(40)]
    )
    if nulls:
        left_ids = [None if rng.random() < 0.1 else i for i in left_ids]
        right_ids = [None if rng.random() < 0.1 else i for i in right_ids]

    if kind == "named":
        left_keys, right_keys = _key_values("left_named", left_ids), _key_values("right_named", right_ids)
        keys: dict[str, Any] = {"left_on": "lk", "right_on": "rk"}
    else:
        left_keys, right_keys = _key_values(kind, left_ids), _key_values(kind, right_ids)
        keys = {"on": list(left_keys)}

    left = pl.LazyFrame({**left_keys, "a": [rng.random() for _ in left_ids]})
    right = pl.LazyFrame(
        {
            **right_keys,
            "b": [rng.randrange(1000) for _ in right_ids],
            "c": [f"c{rng.randrange(1000)}" for _ in right_ids],
        }
    )
    return left, right, keys


@pytest.mark.parametrize("small_restore", [3, 50], indirect=True)
@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("kind", ["int", "str", "int_str", "named"])
@pytest.mark.parametrize("right_unique", [True, False])
@pytest.mark.parametrize("nulls", [False, True])
def test_merge_on_disk_matches_merge(
    tmp_path: Path, how: plh.HOW, kind: str, right_unique: bool, nulls: bool
) -> None:
    """Match ``merge`` for every bucket count, indicator, and column choice."""
    left, right, keys = _frames(kind, right_unique, nulls, seed=len(kind) * 10 + nulls)
    cases = itertools.product([1, 2, 7], ["_merge", None], [None, ["b"]])
    for n, (buckets, indicator, columns) in enumerate(cases):
        expected = plh.merge(
            left, right, columns=columns, how=how, indicator=indicator, **keys
        ).collect()
        got = plh.merge_on_disk(
            left, right, columns=columns, how=how, indicator=indicator, **keys,
            directory=tmp_path / str(n), buckets=buckets,
        ).collect()
        assert got.equals(expected), (buckets, indicator, columns)


@pytest.mark.parametrize("small_restore", [3, 50], indirect=True)
@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("buckets", [1, 4])
def test_merge_on_disk_matches_merge_with_different_key_types(
    tmp_path: Path, how: plh.HOW, buckets: int
) -> None:
    """Put equal Int32 and Int64 keys in one bucket."""
    left = pl.LazyFrame(
        {"k": list(range(40)) * 3, "a": list(range(120))}, schema={"k": pl.Int32, "a": pl.Int64}
    )
    right = pl.LazyFrame(
        {"k": list(range(20, 60)), "b": list(range(40))}, schema={"k": pl.Int64, "b": pl.Int64}
    )
    expected = plh.merge(left, right, on="k", how=how).collect()
    got = plh.merge_on_disk(left, right, on="k", how=how, directory=tmp_path, buckets=buckets).collect()
    assert got.equals(expected)
    assert got.schema == expected.schema


@pytest.mark.parametrize("small_restore", [3, 50], indirect=True)
@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("buckets", [1, 4])
@pytest.mark.parametrize("empty_side", ["left", "right", "both"])
def test_merge_on_disk_with_empty_input(
    tmp_path: Path, how: plh.HOW, buckets: int, empty_side: str
) -> None:
    """Return an empty result with the schema of ``merge``."""
    left = pl.LazyFrame({"k": [1, 2, 3], "a": ["x", "y", "z"]})
    right = pl.LazyFrame({"k": [2, 3, 4], "b": [20, 30, 40]})
    if empty_side in {"left", "both"}:
        left = left.clear()
    if empty_side in {"right", "both"}:
        right = right.clear()
    expected = plh.merge(left, right, on="k", how=how).collect()
    got = plh.merge_on_disk(
        left, right, on="k", how=how, directory=tmp_path, buckets=buckets
    ).collect()
    assert got.equals(expected)
    assert got.schema == expected.schema


def _wide_frame(rows: int) -> pl.LazyFrame:
    """Build a frame of two UInt32 and fifty Float64 columns, 408 bytes a row.

    Parameters
    ----------
    rows : int
        Number of rows.

    Returns
    -------
    pl.LazyFrame
        Frame like the merged rows of a wide table with its row numbers.
    """
    return pl.select(
        *(pl.int_range(rows, dtype=pl.UInt32).alias(f"r{i}") for i in range(2)),
        *(pl.int_range(rows).cast(pl.Float64).alias(f"f{i}") for i in range(50)),
    ).lazy()


@pytest.mark.parametrize(
    ("frame", "restore_bytes", "expected"),
    [
        (_wide_frame(20_000), 408 * 1000, 1000),
        (_wide_frame(20_000), None, 2_631_720),
        (_wide_frame(0), None, merge_on_disk_module._RESTORE_ROWS),
        (_wide_frame(20_000), 1, 1),
        (pl.LazyFrame({"k": list(range(100))}), None, merge_on_disk_module._RESTORE_ROWS),
    ],
    ids=["budget", "default", "empty", "at_least_one", "at_most_restore_rows"],
)
def test_restore_limit(
    monkeypatch: pytest.MonkeyPatch,
    frame: pl.LazyFrame,
    restore_bytes: int | None,
    expected: int,
) -> None:
    """Divide the memory budget by the size of one row of the first rows."""
    if restore_bytes is not None:
        monkeypatch.setattr(merge_on_disk_module, "_RESTORE_BYTES", restore_bytes)
    assert _restore_limit(frame) == expected


@pytest.mark.parametrize(
    ("units", "limit", "expected"),
    [
        ([(0, 10, 4), (10, 20, 4), (20, 30, 4)], 8, [(0, 20), (20, 30)]),
        ([(0, 1, 2), (1, 2, 20), (2, 3, 2)], 8, [(0, 1), (1, 2), (2, 3)]),
        ([(0, 10, 3), (30, 40, 3)], 8, [(0, 40)]),
        ([], 8, []),
    ],
    ids=["group", "too_many", "gap", "empty"],
)
def test_split_ranges(
    units: list[tuple[int, int, int]], limit: int, expected: list[tuple[int, int]]
) -> None:
    """Group neighboring units while their rows fit in one range."""
    assert merge_on_disk_module._split_ranges(units, limit) == expected


def _result_files(directory: Path) -> list[pl.DataFrame]:
    """Read the result files of ``merge_on_disk`` one by one, in order.

    Parameters
    ----------
    directory : Path
        Directory the result was written in.

    Returns
    -------
    list[pl.DataFrame]
        Contents of each result file.
    """
    files = sorted(f for f in directory.iterdir() if _RESULT.fullmatch(f.name))
    return [pl.read_parquet(f) for f in files]


@pytest.mark.parametrize("small_restore", [25], indirect=True)
@pytest.mark.parametrize("how", HOWS)
def test_merge_on_disk_limits_rows_with_duplicate_keys(
    tmp_path: Path, small_restore: int, how: plh.HOW
) -> None:
    """Keep each range within the limit when duplicate keys multiply rows."""
    left = pl.LazyFrame({"k": [i // 10 for i in range(30)], "a": list(range(30))})
    right = pl.LazyFrame({"k": [i // 10 for i in range(30)], "b": list(range(30))})
    expected = plh.merge(left, right, on="k", how=how).collect()
    got = plh.merge_on_disk(left, right, on="k", how=how, directory=tmp_path, buckets=2)
    parts = _result_files(tmp_path)
    assert len(parts) > 1
    assert all(part.height <= small_restore for part in parts)
    assert got.collect().equals(expected)


@pytest.mark.parametrize("small_restore", [25], indirect=True)
def test_merge_on_disk_keeps_one_row_with_many_matches_together(
    tmp_path: Path, small_restore: int
) -> None:
    """Put the rows of one left row in one range even beyond the limit."""
    left = pl.LazyFrame({"k": [0, 1, 2], "a": [10, 11, 12]})
    right = pl.LazyFrame({"k": [1] * 40, "b": list(range(40))})
    expected = plh.merge(left, right, on="k").collect()
    got = plh.merge_on_disk(left, right, on="k", directory=tmp_path, buckets=2)
    parts = _result_files(tmp_path)
    large = [part for part in parts if part.height > small_restore]
    assert len(large) == 1
    assert large[0].height == 40
    assert large[0]["a"].unique().to_list() == [11]
    assert got.collect().equals(expected)


def _string_and_int_keys() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Build frames whose keys are String on the left and Int64 on the right.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame]
        Left and right frames that cannot be joined.
    """
    return pl.LazyFrame({"k": ["1"]}), pl.LazyFrame({"k": [1]})


def _expected_join_error() -> type[BaseException]:
    """Find the exception ``merge`` raises for String and Int64 keys.

    Returns
    -------
    type[BaseException]
        Type of the exception raised by collecting the empty merge.
    """
    left, right = _string_and_int_keys()
    with pytest.raises(Exception) as info:
        plh.merge(left.clear(), right.clear(), on="k").collect()
    return type(info.value)


_FRAME = pl.LazyFrame({"k": [1, 2], "a": [1, 2]})


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda d: plh.merge_on_disk(
                pl.DataFrame({"k": [1]}), _FRAME, on="k", directory=d  # type: ignore[arg-type]
            ),
            TypeError,
            "must both be polars.LazyFrame",
        ),
        (
            lambda d: plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=d, buckets=0),
            ValueError,
            "buckets must be a positive integer, got 0",
        ),
        (
            lambda d: plh.merge_on_disk(
                _FRAME, _FRAME, on="k", left_on="k", right_on="k", directory=d
            ),
            ValueError,
            "pass either `on` or `left_on` and `right_on`, not both",
        ),
        (
            lambda d: plh.merge_on_disk(_FRAME, _FRAME, on="k", indicator="a", directory=d),
            ValueError,
            "indicator column 'a' already exists",
        ),
        (
            lambda d: plh.merge_on_disk(*_string_and_int_keys(), on="k", directory=d),
            _expected_join_error(),
            None,
        ),
    ],
    ids=["dataframe", "buckets", "keys", "indicator", "key_types"],
)
def test_merge_on_disk_rejects_before_writing(
    tmp_path: Path,
    call: Callable[[Path], pl.LazyFrame],
    error: type[BaseException],
    match: str | None,
) -> None:
    """Raise the same errors as ``merge`` and write nothing."""
    directory = tmp_path / "out"
    with pytest.raises(error, match=match):
        call(directory)
    assert not directory.exists() or not any(directory.iterdir())


@pytest.mark.parametrize(
    ("directory", "error", "match"),
    [
        (1, TypeError, "directory must be a str, Path, or None"),
        ("", ValueError, "directory must not be an empty string"),
        ("file", NotADirectoryError, "directory is not a directory"),
    ],
    ids=["type", "empty", "file"],
)
def test_merge_on_disk_rejects_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: Path,
    directory: object,
    error: type[BaseException],
    match: str,
) -> None:
    """Reject a bad ``directory`` before writing anything anywhere."""
    file = tmp_path / "file"
    file.write_text("keep")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(error, match=match):
        plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=directory)  # type: ignore[arg-type]
    assert file.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [file]
    assert list(isolated_cache.iterdir()) == []


def test_merge_on_disk_creates_relative_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cache: Path
) -> None:
    """Create a missing relative ``directory`` under the working directory."""
    monkeypatch.chdir(tmp_path)
    out = plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=Path("a") / "b", buckets=2)
    assert out.collect().equals(plh.merge(_FRAME, _FRAME, on="k").collect())
    assert [p.name for p in tmp_path.iterdir()] == ["a"]
    written = [p.name for p in (tmp_path / "a" / "b").iterdir()]
    assert len(written) == 1 and _RESULT.fullmatch(written[0])
    assert list(isolated_cache.iterdir()) == []


def test_merge_on_disk_leaves_other_files_alone(tmp_path: Path) -> None:
    """Neither read nor overwrite files already in ``directory``."""
    placed = {"merge-x-00000.parquet": pl.DataFrame({"z": [9]}), "other.parquet": pl.DataFrame({"k": [5]})}
    for name, frame in placed.items():
        frame.write_parquet(tmp_path / name)
    before = {name: (tmp_path / name).read_bytes() for name in placed}

    first_left, first_right, keys = _frames("int", True, False, seed=1)
    second_left, second_right, _ = _frames("int", False, True, seed=2)
    first = plh.merge_on_disk(first_left, first_right, **keys, directory=tmp_path, buckets=3)
    second = plh.merge_on_disk(second_left, second_right, **keys, directory=tmp_path, buckets=3)

    assert first.collect().equals(plh.merge(first_left, first_right, **keys).collect())
    assert second.collect().equals(plh.merge(second_left, second_right, **keys).collect())
    assert {name: (tmp_path / name).read_bytes() for name in placed} == before
    assert not list(tmp_path.glob("*-work"))


_WIN_HOME = ("home", "AppData", "Local", "polars-helpers", "cache")
_POSIX_HOME = ("home", ".cache", "polars-helpers")


@pytest.mark.parametrize(
    ("platform", "variable", "value", "expected"),
    [
        ("win32", "LOCALAPPDATA", "local", ("local", "polars-helpers", "cache")),
        ("win32", "LOCALAPPDATA", None, _WIN_HOME),
        ("win32", "LOCALAPPDATA", "", _WIN_HOME),
        ("win32", "LOCALAPPDATA", "relative", _WIN_HOME),
        ("darwin", "XDG_CACHE_HOME", "xdg", ("home", "Library", "Caches", "polars-helpers")),
        ("linux", "XDG_CACHE_HOME", "xdg", ("xdg", "polars-helpers")),
        ("linux", "XDG_CACHE_HOME", None, _POSIX_HOME),
        ("linux", "XDG_CACHE_HOME", "", _POSIX_HOME),
        ("linux", "XDG_CACHE_HOME", "relative", _POSIX_HOME),
    ],
    ids=[
        "win-set", "win-unset", "win-empty", "win-relative", "mac",
        "posix-set", "posix-unset", "posix-empty", "posix-relative",
    ],
)
def test_merge_on_disk_writes_in_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    variable: str,
    value: str | None,
    expected: tuple[str, ...],
) -> None:
    """Write directly in the default root of ``_disk``, never in ``tempfile``.

    ``sys.platform`` itself is not patched, since the locks depend on it;
    the default root is found for ``platform`` from the patched environment.
    """
    assert not hasattr(merge_on_disk_module, "tempfile")
    monkeypatch.chdir(tmp_path)
    if value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, str(tmp_path / value) if value in {"local", "xdg"} else value)
    monkeypatch.setattr(
        _disk,
        "default_root",
        lambda: _disk._default_root_for(platform, os.environ, lambda: tmp_path / "home"),
    )
    out = plh.merge_on_disk(_FRAME, _FRAME, on="k", buckets=2, delete_on_exit=False)
    assert out.collect().equals(plh.merge(_FRAME, _FRAME, on="k").collect())
    written = [p.name for p in tmp_path.joinpath(*expected).iterdir()]
    assert len(written) == 1 and _RESULT.fullmatch(written[0])
    assert [p.name for p in tmp_path.iterdir()] == [expected[0]]


def _child_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Build the environment of a child whose default root is in ``tmp_path``.

    Parameters
    ----------
    tmp_path : Path
        Directory of the test.

    Returns
    -------
    tuple[dict[str, str], Path]
        Environment, with ``LOCALAPPDATA`` for Windows and ``HOME`` and
        ``XDG_CACHE_HOME`` for POSIX under ``tmp_path / "home"``, and the
        default root the child sees with it.
    """
    home = tmp_path / "home"
    env = {
        **os.environ,
        "LOCALAPPDATA": str(home / "AppData" / "Local"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "HOME": str(home),
        "USERPROFILE": str(home),
    }
    if sys.platform == "win32":
        root = home / "AppData" / "Local" / "polars-helpers" / "cache"
    elif sys.platform == "darwin":
        root = home / "Library" / "Caches" / "polars-helpers"
    else:
        root = home / ".cache" / "polars-helpers"
    return env, root


def _run_child(tmp_path: Path, arguments: str) -> Path:
    """Call ``merge_on_disk`` in a child process that then exits normally.

    Parameters
    ----------
    tmp_path : Path
        Directory of the test; the child sees its default root under it.
    arguments : str
        Keyword arguments for ``directory`` and ``delete_on_exit``.

    Returns
    -------
    Path
        Default root of the child.
    """
    env, root = _child_env(tmp_path)
    code = textwrap.dedent(
        f"""
        import polars as pl

        import polars_helpers as plh

        frame = pl.LazyFrame({{"k": [1, 2, 3], "a": [1, 2, 3]}})
        plh.merge_on_disk(frame, frame, on="k", buckets=2, {arguments}).collect()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    return root


def test_merge_on_disk_keeps_given_directory_by_default(tmp_path: Path) -> None:
    """Keep only the result files in a given directory after the process exits."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    _run_child(tmp_path, f"directory={str(out)!r}")
    names = sorted(p.name for p in out.iterdir())
    assert len(names) == 2 and names[0] == "keep.txt" and _RESULT.fullmatch(names[1])


@pytest.mark.parametrize("created", [False, True], ids=["existing", "created"])
def test_merge_on_disk_deletes_from_given_directory(tmp_path: Path, created: bool) -> None:
    """Delete this call's files, session files, and created directories on exit."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    directory = out / "new" / "sub" if created else out
    _run_child(tmp_path, f"directory={str(directory)!r}, delete_on_exit=True")
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


@pytest.mark.parametrize("existing", [False, True], ids=["new", "existing"])
def test_merge_on_disk_deletes_from_default_root(tmp_path: Path, existing: bool) -> None:
    """Delete this call's files and session files but keep the default root."""
    _, root = _child_env(tmp_path)
    if existing:
        root.mkdir(parents=True)
        (root / "keep.txt").write_text("keep")
    _run_child(tmp_path, "directory=None")
    assert root.is_dir()
    assert sorted(p.name for p in root.iterdir()) == (["keep.txt"] if existing else [])


def test_merge_on_disk_keeps_result_in_default_root(tmp_path: Path) -> None:
    """Keep only the result files in the default root with ``delete_on_exit=False``."""
    root = _run_child(tmp_path, "directory=None, delete_on_exit=False")
    names = [p.name for p in root.iterdir()]
    assert len(names) == 1 and _RESULT.fullmatch(names[0])


# Child that calls ``merge_on_disk`` in ``sys.argv[1]``, the default root if
# ``sys.argv[3]`` is ``"1"``, with and without ``delete_on_exit``, and in
# ``sys.argv[2]`` with it, then prints the names each call added.
_MERGING_CHILD = """
import json
from pathlib import Path

import polars as pl

import polars_helpers as plh

root, other = Path(sys.argv[1]), Path(sys.argv[2])
directory = None if sys.argv[3] == "1" else root
frame = pl.LazyFrame({"k": [1, 2, 3], "a": [1, 2, 3]})
names = {}
for name, where, delete in (("delete", root, True), ("keep", root, False), ("other", other, True)):
    before = set(os.listdir(where))
    plh.merge_on_disk(
        frame, frame, on="k", buckets=2, delete_on_exit=delete,
        directory=directory if where == root else where,
    ).collect()
    names[name] = sorted(set(os.listdir(where)) - before)
print(json.dumps(names), flush=True)
wait_for_parent()
"""


@pytest.mark.parametrize("use_default", [False, True], ids=["directory", "default"])
@pytest.mark.parametrize("collector", ["merge_delete", "merge_keep", "scan_parquet_relaxed"])
def test_merge_on_disk_collects_killed_process(
    tmp_path: Path, isolated_cache: Path, use_default: bool, collector: str
) -> None:
    """Collect what a killed process registered on the next call on its root."""
    root = _disk.default_root() if use_default else tmp_path / "root"
    assert root.is_relative_to(isolated_cache if use_default else tmp_path)
    other = tmp_path / "other"
    root.mkdir(parents=True)
    other.mkdir()
    (root / "keep.txt").write_text("keep")
    pl.DataFrame({"z": [9]}).write_parquet(root / "merge-unregistered-00000.parquet")
    arguments = (str(root), str(other), "1" if use_default else "0")

    alive = helpers.start(_MERGING_CHILD, *arguments)
    try:
        alive_names = json.loads(helpers.ready(alive))
        killed = helpers.start(_MERGING_CHILD, *arguments)
        killed_names = json.loads(helpers.ready(killed))
        helpers.kill(killed)
        assert len(killed_names["delete"]) == 3
        assert any(n.startswith(_disk.SESSION_PREFIX) for n in killed_names["delete"])
        assert len(killed_names["keep"]) == 1 and _RESULT.fullmatch(killed_names["keep"][0])

        directory = None if use_default else root
        if collector == "scan_parquet_relaxed":
            source = tmp_path / "source.parquet"
            pl.DataFrame({"x": [1]}).write_parquet(source)
            plh.scan_parquet_relaxed(source, directory=directory).collect()
        else:
            plh.merge_on_disk(
                _FRAME, _FRAME, on="k", directory=directory,
                delete_on_exit=collector == "merge_delete",
            ).collect()

        names = {p.name for p in root.iterdir()}
        assert not names & set(killed_names["delete"])
        assert set(killed_names["keep"]) <= names
        assert {*alive_names["delete"], *alive_names["keep"]} <= names
        assert {"keep.txt", "merge-unregistered-00000.parquet"} <= names
        assert {*killed_names["other"], *alive_names["other"]} <= {p.name for p in other.iterdir()}
    finally:
        helpers.finish(alive)
    names = {p.name for p in root.iterdir()}
    assert not names & set(alive_names["delete"])
    assert set(alive_names["keep"]) <= names


@pytest.mark.parametrize("delete_on_exit", [None, True, False])
@pytest.mark.parametrize("use_default", [False, True], ids=["directory", "default"])
def test_merge_on_disk_cleans_up_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache: Path,
    delete_on_exit: bool | None,
    use_default: bool,
) -> None:
    """Delete the files and directories it wrote when the merge fails midway."""
    calls = 0
    real_merge = merge_on_disk_module.merge

    def failing_merge(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("boom")
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(merge_on_disk_module, "merge", failing_merge)
    left, right, keys = _frames("int", True, False, seed=3)
    (tmp_path / "keep.txt").write_text("keep")
    directory = None if use_default else tmp_path / "out" / "sub"
    with pytest.raises(RuntimeError, match="boom"):
        plh.merge_on_disk(
            left, right, **keys, directory=directory, delete_on_exit=delete_on_exit, buckets=4
        )
    assert calls == 2
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]
    assert list(isolated_cache.iterdir()) == []
    root = _disk.default_root() if use_default else tmp_path / "out" / "sub"
    assert _disk.current_session_id(root) is None


def _fail_saves_after(monkeypatch: pytest.MonkeyPatch, allowed: int) -> None:
    """Make the JSON file of a session fail to be written after some writes.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch ``Session._save``.
    allowed : int
        Number of writes that succeed.
    """
    real_save = _disk.Session._save
    saves = 0

    def save(self: _disk.Session) -> None:
        nonlocal saves
        saves += 1
        if saves > allowed:
            raise OSError("disk full")
        real_save(self)

    monkeypatch.setattr(_disk.Session, "_save", save)


@pytest.mark.parametrize("use_default", [False, True], ids=["directory", "default"])
def test_merge_on_disk_cleans_up_after_failed_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cache: Path, use_default: bool
) -> None:
    """Delete the session it created when registering the result fails."""
    (tmp_path / "keep.txt").write_text("keep")
    directory = None if use_default else tmp_path / "out" / "sub"
    root = _disk.default_root() if use_default else tmp_path / "out" / "sub"
    _fail_saves_after(monkeypatch, 1)
    with pytest.raises(OSError, match="disk full"):
        plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=directory, delete_on_exit=True)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]
    assert list(isolated_cache.iterdir()) == []
    assert _disk.current_session_id(root) is None


def test_merge_on_disk_keeps_existing_session_after_failed_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the session and the earlier result when registering a later one fails."""
    first = plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=tmp_path, delete_on_exit=True)
    session_id = _disk.current_session_id(tmp_path)
    assert session_id is not None
    first_names = {p.name for p in tmp_path.iterdir()}
    real_save = _disk.Session._save
    _fail_saves_after(monkeypatch, 0)
    with pytest.raises(OSError, match="disk full"):
        plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=tmp_path, delete_on_exit=True)
    assert {p.name for p in tmp_path.iterdir()} == first_names
    assert _disk.current_session_id(tmp_path) == session_id
    data = _disk.read_json(tmp_path / f"{_disk.SESSION_PREFIX}{session_id}.json")
    assert data is not None
    assert [Path(f).name for f in data["files"]] == [  # type: ignore[union-attr]
        n for n in first_names if _RESULT.fullmatch(n)
    ]
    assert first.collect().equals(plh.merge(_FRAME, _FRAME, on="k").collect())

    monkeypatch.setattr(_disk.Session, "_save", real_save)
    plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=tmp_path, delete_on_exit=True)
    data = _disk.read_json(tmp_path / f"{_disk.SESSION_PREFIX}{session_id}.json")
    assert data is not None and len(data["files"]) == 2  # type: ignore[arg-type]


@pytest.mark.parametrize("indicator", ["_merge", None])
@pytest.mark.parametrize("buckets", [1, 4])
def test_merge_on_disk_verbose_matches_merge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], indicator: str | None, buckets: int
) -> None:
    """Print the same summary as ``merge(..., verbose=True)``."""
    left, right, keys = _frames("int", False, True, seed=4)
    plh.merge(left, right, **keys, how="full", indicator=indicator, verbose=True)
    expected = capsys.readouterr().out
    plh.merge_on_disk(
        left, right, **keys, how="full", indicator=indicator, verbose=True,
        directory=tmp_path, buckets=buckets,
    )
    assert capsys.readouterr().out == expected
    assert expected


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize(
    ("left_extra", "right_extra", "indicator"),
    [
        ("__polars_helpers_bucket", None, "_merge"),
        ("__polars_helpers_left_row", None, "_merge"),
        ("__polars_helpers_right_row", None, "_merge"),
        (None, "__polars_helpers_left_row", "_merge"),
        ("x", "x", "__polars_helpers_left_row"),
        ("__polars_helpers_left_row", "__polars_helpers_left_row", "__polars_helpers_right_row"),
    ],
    ids=["bucket", "left_row", "right_row_in_left", "left_row_in_right", "indicator", "suffix"],
)
def test_merge_on_disk_with_columns_named_like_temporary_ones(
    tmp_path: Path,
    how: plh.HOW,
    left_extra: str | None,
    right_extra: str | None,
    indicator: str,
) -> None:
    """Keep input columns and indicators named like the temporary columns."""
    left = pl.LazyFrame({"k": [1, 2, 2, 3, None], "a": [1, 2, 3, 4, 5]})
    right = pl.LazyFrame({"k": [2, 3, 4, None], "b": [20, 30, 40, 50]})
    if left_extra is not None:
        left = left.with_columns(pl.int_range(pl.len()).alias(left_extra) * 10)
    if right_extra is not None:
        right = right.with_columns(pl.int_range(pl.len()).alias(right_extra) * 100)
    expected = plh.merge(left, right, on="k", how=how, indicator=indicator).collect()
    got = plh.merge_on_disk(
        left, right, on="k", how=how, indicator=indicator, directory=tmp_path, buckets=4
    ).collect()
    assert got.equals(expected)


def test_merge_on_disk_keeps_relative_directory_after_chdir(tmp_path: Path) -> None:
    """Read and delete the result of a relative ``directory`` after ``chdir``."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    code = textwrap.dedent(
        f"""
        import os

        import polars as pl

        import polars_helpers as plh

        os.chdir({str(first)!r})
        frame = pl.LazyFrame({{"k": [1, 2, 3], "a": [1, 2, 3]}})
        for buckets in (1, 2):
            out = plh.merge_on_disk(
                frame, frame, on="k", directory="out", delete_on_exit=True, buckets=buckets
            )
            os.chdir({str(second)!r})
            assert out.collect().equals(plh.merge(frame, frame, on="k").collect())
            os.chdir({str(first)!r})
        os.chdir({str(second)!r})
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert list(first.iterdir()) == []
    assert list(second.iterdir()) == []
