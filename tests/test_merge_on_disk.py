"""Tests for merging two LazyFrames on disk with ``merge_on_disk``."""

from __future__ import annotations

import importlib
import itertools
import random
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import polars_helpers as plh

merge_on_disk_module = importlib.import_module("polars_helpers.merge_on_disk")

HOWS: list[plh.HOW] = ["left", "inner", "right", "full"]


@pytest.fixture(autouse=True)
def small_restore_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the rows back in order three row numbers at a time.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to shrink ``_RESTORE_ROWS``.
    """
    monkeypatch.setattr(merge_on_disk_module, "_RESTORE_ROWS", 3)


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


def test_merge_on_disk_rejects_file_as_directory(tmp_path: Path) -> None:
    """Raise ``NotADirectoryError`` for a file passed as ``directory``."""
    file = tmp_path / "file"
    file.write_text("keep")
    with pytest.raises(NotADirectoryError):
        plh.merge_on_disk(_FRAME, _FRAME, on="k", directory=file)
    assert file.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [file]


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


def _run_child(tmp_path: Path, arguments: str, temp_dir: bool) -> None:
    """Call ``merge_on_disk`` in a child process that then exits normally.

    Parameters
    ----------
    tmp_path : Path
        Directory of the test.
    arguments : str
        Keyword arguments for ``directory`` and ``delete_on_exit``.
    temp_dir : bool
        Whether to point ``tempfile.tempdir`` of the child at ``tmp_path``.
    """
    code = textwrap.dedent(
        f"""
        import tempfile
        from pathlib import Path

        import polars as pl

        import polars_helpers as plh

        if {temp_dir!r}:
            tempfile.tempdir = {str(tmp_path)!r}
        frame = pl.LazyFrame({{"k": [1, 2, 3], "a": [1, 2, 3]}})
        plh.merge_on_disk(frame, frame, on="k", buckets=2, {arguments}).collect()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_merge_on_disk_keeps_given_directory_by_default(tmp_path: Path) -> None:
    """Keep the result files in a given directory after the process exits."""
    (tmp_path / "keep.txt").write_text("keep")
    _run_child(tmp_path, f"directory={str(tmp_path)!r}", temp_dir=False)
    assert list(tmp_path.glob("merge-*-00000.parquet"))
    assert (tmp_path / "keep.txt").exists()


def test_merge_on_disk_deletes_from_given_directory(tmp_path: Path) -> None:
    """Delete only this call's files from a given directory on exit."""
    (tmp_path / "keep.txt").write_text("keep")
    _run_child(tmp_path, f"directory={str(tmp_path)!r}, delete_on_exit=True", temp_dir=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_merge_on_disk_deletes_temporary_directory_by_default(tmp_path: Path) -> None:
    """Delete the directory it created when ``directory`` is ``None``."""
    (tmp_path / "keep.txt").write_text("keep")
    _run_child(tmp_path, "directory=None", temp_dir=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_merge_on_disk_keeps_temporary_directory(tmp_path: Path) -> None:
    """Keep the created directory with ``delete_on_exit=False``."""
    (tmp_path / "keep.txt").write_text("keep")
    _run_child(tmp_path, "directory=None, delete_on_exit=False", temp_dir=True)
    created = list(tmp_path.glob("polars_helpers_merge_*"))
    assert len(created) == 1
    assert list(created[0].glob("merge-*-00000.parquet"))
    assert (tmp_path / "keep.txt").exists()


def test_merge_on_disk_cleans_up_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete the files it wrote when the merge fails midway."""
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
    with pytest.raises(RuntimeError, match="boom"):
        plh.merge_on_disk(left, right, **keys, directory=tmp_path, buckets=4)
    assert calls == 2
    assert not list(tmp_path.glob("merge-*"))


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
