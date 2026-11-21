"""Tests for the parts of ``scan_parquet_relaxed`` kept on disk.

They cover reuse, the time limit, changes of the inputs, the protection of
parts in use, deletion on exit, collection of what killed processes left,
and concurrent calls.
"""

import gc
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import helpers
import polars as pl
import pytest

import polars_helpers as plh
from polars_helpers import _disk, _parquet_cache, parquet
from polars_helpers._parquet_cache import Column

DAY = 86400

CALL_CHILD = """
import json
import polars_helpers as plh
args = json.loads(sys.argv[1])
lf = plh.scan_parquet_relaxed(
    args["paths"],
    directory=args["directory"],
    cache_ttl_days=args["ttl"],
    columns=args.get("columns"),
)
print(json.dumps(lf.collect().to_dicts()), flush=True)
wait_for_parent()
"""


def call_args(paths: list[Path], directory: Path | str | None, ttl: int | None) -> str:
    """Encode the arguments of ``CALL_CHILD``.

    Parameters
    ----------
    paths : list[Path]
        Source files.
    directory : Path | str | None
        Directory of the parts.
    ttl : int | None
        ``cache_ttl_days``.

    Returns
    -------
    str
        JSON of the arguments.
    """
    return json.dumps(
        {
            "paths": [str(p) for p in paths],
            "directory": None if directory is None else str(directory),
            "ttl": ttl,
        }
    )


def write(path: Path, frame: pl.DataFrame) -> Path:
    """Write a frame to a Parquet file.

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


@pytest.fixture
def sources(tmp_path: Path) -> list[Path]:
    """Write two source files with different columns.

    Parameters
    ----------
    tmp_path : Path
        Temporary folder of the test.

    Returns
    -------
    list[Path]
        The files.
    """
    return [
        write(tmp_path / "src" / "a.parquet", pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})),
        write(tmp_path / "src" / "b.parquet", pl.DataFrame({"x": [3], "z": [True]})),
    ]


EXPECTED = [
    {"x": 1, "y": "a", "z": None},
    {"x": 2, "y": "b", "z": None},
    {"x": 3, "y": None, "z": True},
]


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record the parts written in this process.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patcher.

    Returns
    -------
    list[tuple[str, str]]
        Name of the source file and column of each part written.
    """
    log: list[tuple[str, str]] = []
    real = parquet._part_writer

    def recording(path: Path, schema: pl.Schema) -> Callable[[Column, Path], None]:
        inner = real(path, schema)

        def write_part(column: Column, out: Path) -> None:
            log.append((path.name, column.name))
            inner(column, out)

        return write_part

    monkeypatch.setattr(parquet, "_part_writer", recording)
    return log


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the clock of the parts by one the test sets.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patcher.

    Returns
    -------
    list[float]
        One-item list holding the current time, starting now.
    """
    now = [time.time()]
    monkeypatch.setattr(_parquet_cache, "_now", lambda: now[0])
    return now


def entries(root: Path) -> list[Path]:
    """List the entry directories of shared parts in a root.

    Parameters
    ----------
    root : Path
        Root.

    Returns
    -------
    list[Path]
        Entry directories, sorted.
    """
    return sorted(p for p in root.glob("polars-helpers-parquet-*-ttl*-*") if p.is_dir())


def generations(root: Path) -> list[Path]:
    """List the generation directories of shared parts in a root.

    Parameters
    ----------
    root : Path
        Root.

    Returns
    -------
    list[Path]
        Generation directories, sorted.
    """
    return sorted(g for e in entries(root) for g in e.iterdir() if g.is_dir())


def metas(root: Path) -> list[dict[str, object]]:
    """Read the metadata of every generation in a root.

    Parameters
    ----------
    root : Path
        Root.

    Returns
    -------
    list[dict[str, object]]
        Metadata, in the order of ``generations``.
    """
    return [json.loads((g / "meta.json").read_text(encoding="utf-8")) for g in generations(root)]


def session_files(root: Path) -> list[Path]:
    """List the session files in a root.

    Parameters
    ----------
    root : Path
        Root.

    Returns
    -------
    list[Path]
        Session files.
    """
    return sorted(root.glob(".polars-helpers-session-*"))


# ----------------------------------------------------------------- reuse


def test_none_reuses_parts_in_the_process(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Reuse the private parts in the same process."""
    root = tmp_path / "cache"
    first = plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=None)
    assert len(writes) == 6
    second = plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=None)
    assert len(writes) == 6
    assert first.collect().to_dicts() == second.collect().to_dicts() == EXPECTED
    assert entries(root) == []
    parts = sorted(p.name for p in root.glob("*.parquet"))
    assert len(parts) == 6
    assert all(p.startswith("polars-helpers-parquet-") for p in parts)


def test_ttl_namespaces_are_separate(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Keep parts of different time limits apart."""
    root = tmp_path / "cache"
    plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=1)
    one = metas(root)
    plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=None)
    plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=7)
    assert len(writes) == 18
    assert len(entries(root)) == 4
    seven = [m for m in metas(root) if m["ttl_days"] == 7]
    assert len(seven) == 2
    assert [m for m in metas(root) if m["ttl_days"] == 1] == one


def test_ttl_reuses_parts_of_another_process(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Reuse the shared parts a finished process left."""
    root = tmp_path / "cache"
    out = helpers.run(CALL_CHILD, call_args(sources, root, 1))
    assert json.loads(out) == EXPECTED
    assert session_files(root) == []
    assert not list(root.rglob("lease-*"))
    result = plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=1)
    assert writes == []
    assert result.collect().to_dicts() == EXPECTED


def test_columns_reuse_writes_only_new_columns(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Write only the parts of columns that were not prepared yet."""
    root = tmp_path / "cache"
    plh.scan_parquet_relaxed(sources, directory=root, columns=["x"])
    assert writes == [("a.parquet", "x"), ("b.parquet", "x")]
    writes.clear()
    out = plh.scan_parquet_relaxed(sources, directory=root, columns=["y", "x"])
    assert writes == [("a.parquet", "y"), ("b.parquet", "y")]
    assert out.collect().to_dicts() == [
        {"y": "a", "x": 1},
        {"y": "b", "x": 2},
        {"y": None, "x": 3},
    ]
    stored = {
        tuple(pl.read_parquet_schema(p)) for p in root.rglob("*.parquet")
    }
    assert stored == {("x",), ("y",)}


def test_roots_do_not_share_parts(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Prepare parts again in another root."""
    plh.scan_parquet_relaxed(sources, directory=tmp_path / "one")
    plh.scan_parquet_relaxed(sources, directory=tmp_path / "two")
    assert len(writes) == 12


# ------------------------------------------------------------ time limit


def test_time_limit_is_last_use_plus_days(
    tmp_path: Path,
    sources: list[Path],
    writes: list[tuple[str, str]],
    clock: list[float],
) -> None:
    """Expire exactly at the last use plus 24 hours, extended by reuse."""
    root = tmp_path / "cache"
    start = clock[0]
    plh.scan_parquet_relaxed(sources, directory=root)
    assert {m["last_used_at"] for m in metas(root)} == {start}
    assert len(writes) == 6

    clock[0] = start + DAY - 0.001
    plh.scan_parquet_relaxed(sources, directory=root)
    assert len(writes) == 6
    assert {m["last_used_at"] for m in metas(root)} == {start + DAY - 0.001}
    assert {m["created_at"] for m in metas(root)} == {start}

    clock[0] = start + 2 * DAY - 0.002
    plh.scan_parquet_relaxed(sources, directory=root)
    assert len(writes) == 6

    clock[0] = start + 3 * DAY - 0.002
    plh.scan_parquet_relaxed(sources, directory=root)
    assert len(writes) == 12
    assert len(generations(root)) == 4  # the old ones are still leased


def test_several_days(
    tmp_path: Path,
    sources: list[Path],
    writes: list[tuple[str, str]],
    clock: list[float],
) -> None:
    """Keep parts for the given number of days."""
    root = tmp_path / "cache"
    start = clock[0]
    plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=7)
    clock[0] = start + 7 * DAY - 1
    plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=7)
    assert len(writes) == 6
    clock[0] = start + 14 * DAY - 1
    plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=7)
    assert len(writes) == 12


def test_collect_does_not_extend(
    tmp_path: Path, sources: list[Path], clock: list[float]
) -> None:
    """Extend the time limit only by calls of the function."""
    root = tmp_path / "cache"
    start = clock[0]
    out = plh.scan_parquet_relaxed(sources, directory=root)
    clock[0] = start + 100
    out.collect()
    out.select("x").collect()
    assert {m["last_used_at"] for m in metas(root)} == {start}


def test_zero_days_reuses_nothing(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Prepare new parts on every call, keeping those in use."""
    root = tmp_path / "cache"
    first = plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=0)
    second = plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=0)
    assert len(writes) == 12
    assert len(generations(root)) == 4
    assert first.collect().to_dicts() == second.collect().to_dicts() == EXPECTED


@pytest.mark.parametrize(
    ("ttl", "error"),
    [(-1, ValueError), (True, TypeError), (1.0, TypeError), ("1", TypeError)],
)
def test_invalid_ttl(
    tmp_path: Path, sources: list[Path], ttl: object, error: type[Exception]
) -> None:
    """Reject a negative or non-integer time limit before writing."""
    root = tmp_path / "cache"
    with pytest.raises(error, match="cache_ttl_days"):
        plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=ttl)  # type: ignore[arg-type]
    assert not root.exists()


# --------------------------------------------------------- input changes


def test_changed_input_is_prepared_again(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Prepare a changed file again; older frames still read the old data."""
    root = tmp_path / "cache"
    old = plh.scan_parquet_relaxed(sources, directory=root)
    derived = old.filter(pl.col("x") > 1).select("x", "y")
    write(sources[0], pl.DataFrame({"x": [10, 20, 30], "y": ["p", "q", "r"]}))
    writes.clear()
    new = plh.scan_parquet_relaxed(sources, directory=root)
    assert sorted(writes) == [("a.parquet", "x"), ("a.parquet", "y"), ("a.parquet", "z")]
    assert new.collect()["x"].to_list() == [10, 20, 30, 3]
    assert old.collect().to_dicts() == EXPECTED
    assert derived.collect().to_dicts() == [{"x": 2, "y": "b"}, {"x": 3, "y": None}]


def test_common_type_change_prepares_affected_columns(
    tmp_path: Path, sources: list[Path], writes: list[tuple[str, str]]
) -> None:
    """Prepare again only the columns whose common type changed."""
    root = tmp_path / "cache"
    plh.scan_parquet_relaxed(sources, directory=root)
    wider = write(tmp_path / "src" / "c.parquet", pl.DataFrame({"x": [0.5]}))
    writes.clear()
    out = plh.scan_parquet_relaxed([*sources, wider], directory=root)
    assert sorted(writes) == [
        ("a.parquet", "x"),
        ("b.parquet", "x"),
        ("c.parquet", "x"),
        ("c.parquet", "y"),
        ("c.parquet", "z"),
    ]
    assert out.collect()["x"].to_list() == [1.0, 2.0, 3.0, 0.5]


# ---------------------------------------------------- failures and leftovers


def test_failed_write_keeps_complete_parts(
    tmp_path: Path, sources: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail without touching complete parts or leaving unfinished ones."""
    root = tmp_path / "cache"
    first = plh.scan_parquet_relaxed(sources, directory=root, columns=["x"])
    before = metas(root)
    real = parquet._part_writer

    def failing(path: Path, schema: pl.Schema) -> Callable[[Column, Path], None]:
        inner = real(path, schema)

        def write_part(column: Column, out: Path) -> None:
            inner(column, out)
            raise OSError("disk full")

        return write_part

    monkeypatch.setattr(parquet, "_part_writer", failing)
    with pytest.raises(OSError, match="disk full"):
        plh.scan_parquet_relaxed(sources, directory=root, columns=["x", "y"])
    assert not list(root.rglob("*.tmp"))
    assert metas(root) == before
    assert first.collect()["x"].to_list() == [1, 2, 3]
    monkeypatch.setattr(parquet, "_part_writer", real)
    out = plh.scan_parquet_relaxed(sources, directory=root, columns=["x", "y"])
    assert out.collect()["y"].to_list() == ["a", "b", None]


def test_unfinished_writes_are_collected(tmp_path: Path, sources: list[Path]) -> None:
    """Delete generations and parts that a write left unfinished."""
    root = tmp_path / "cache"
    plh.scan_parquet_relaxed(sources, directory=root)
    entry = entries(root)[0]
    complete = next(g for g in entry.iterdir() if g.is_dir())
    unfinished = entry / "0123456789ab"
    unfinished.mkdir()
    (unfinished / "aaaaaaaaaaaa.1234abcd.tmp").write_bytes(b"x")
    (complete / "bbbbbbbbbbbb.parquet").write_bytes(b"x")
    (complete / "cccccccccccc.9876fedc.tmp").write_bytes(b"x")
    unrelated = entry / "notes.txt"
    unrelated.write_text("x")
    plh.scan_parquet_relaxed(sources[1], directory=root)
    assert not unfinished.exists()
    assert not (complete / "bbbbbbbbbbbb.parquet").exists()
    assert not (complete / "cccccccccccc.9876fedc.tmp").exists()
    assert unrelated.exists()
    assert len(list(complete.glob("*.parquet"))) == 3


def test_other_files_and_users_are_kept(
    tmp_path: Path, sources: list[Path], clock: list[float]
) -> None:
    """Never delete files of other programs or entries of other users."""
    root = tmp_path / "cache"
    root.mkdir()
    (root / "notes.txt").write_text("x")
    (root / "keep").mkdir()
    (root / "keep" / "data.parquet").write_bytes(b"x")
    theirs = root / "polars-helpers-parquet-deadbeef-ttl1-0123456789abcdef" / "0123456789ab"
    theirs.mkdir(parents=True)
    (theirs / "meta.json").write_text(
        json.dumps({"marker": "polars-helpers-parquet", "user": "someone-else"})
    )
    ours_by_name = root / f"{_parquet_cache._entry_name(1, _parquet_cache.Source(Path('x'), 1, 1))}"
    other_gen = ours_by_name / "abcdefabcdef"
    other_gen.mkdir(parents=True)
    (other_gen / "meta.json").write_text(
        json.dumps(
            {
                "marker": "polars-helpers-parquet",
                "format": 1,
                "user": "someone-else",
                "ttl_days": 1,
                "last_used_at": 0,
                "parts": {},
            }
        )
    )
    plh.scan_parquet_relaxed(sources, directory=root)
    clock[0] += 10 * DAY
    plh.scan_parquet_relaxed(sources, directory=root)
    assert (root / "notes.txt").exists()
    assert (root / "keep" / "data.parquet").exists()
    assert (theirs / "meta.json").exists()
    assert (other_gen / "meta.json").exists()


# ------------------------------------------------- protection and exit


def test_parts_in_use_survive_expiry(
    tmp_path: Path, sources: list[Path], clock: list[float]
) -> None:
    """Keep the parts a frame of this process reads, even when expired."""
    root = tmp_path / "cache"
    lf = plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=0)
    derived = lf.select("x").filter(pl.col("x") >= 2)
    lf.collect()
    del lf
    gc.collect()
    clock[0] += 5 * DAY
    other = write(tmp_path / "src2" / "c.parquet", pl.DataFrame({"x": [9]}))
    plh.scan_parquet_relaxed(other, directory=root, cache_ttl_days=0)
    plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=0)
    assert derived.collect()["x"].to_list() == [2, 3]


def test_expired_parts_of_ended_processes_are_collected(
    tmp_path: Path, sources: list[Path], clock: list[float]
) -> None:
    """Delete expired parts nobody uses at the next call on the root."""
    root = tmp_path / "cache"
    helpers.run(CALL_CHILD, call_args(sources, root, 1))
    other = write(tmp_path / "src2" / "c.parquet", pl.DataFrame({"x": [9]}))
    plh.scan_parquet_relaxed(other, directory=root)
    assert len(entries(root)) == 3
    clock[0] += DAY + 60
    plh.scan_parquet_relaxed(other, directory=root)
    assert len(entries(root)) == 1
    assert sorted(p.name for p in root.glob("*.lock") if "parquet" in p.name) == [
        f"{entries(root)[0].name}.lock"
    ]


def test_parts_of_living_processes_are_kept(
    tmp_path: Path, sources: list[Path], clock: list[float]
) -> None:
    """Keep expired parts a living process reads; collect them once it is killed."""
    root = tmp_path / "cache"
    proc = helpers.start(CALL_CHILD, call_args(sources, root, 1))
    try:
        assert json.loads(helpers.ready(proc)) == EXPECTED
        clock[0] += 3 * DAY
        other = write(tmp_path / "src2" / "c.parquet", pl.DataFrame({"x": [9]}))
        plh.scan_parquet_relaxed(other, directory=root)
        assert len(entries(root)) == 3
        assert len(list(root.rglob("lease-*"))) == 3
    finally:
        helpers.kill(proc)
    plh.scan_parquet_relaxed(other, directory=root)
    assert len(entries(root)) == 1
    assert len(session_files(root)) == 2  # only this process


def test_none_child_deletes_its_parts_on_exit(tmp_path: Path) -> None:
    """Delete private parts and the directories made for them on exit."""
    source = write(tmp_path / "src" / "a.parquet", pl.DataFrame({"x": [1]}))
    helpers.run(CALL_CHILD, call_args([source], "made/nested/cache", None), cwd=tmp_path)
    assert not (tmp_path / "made").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    helpers.run(CALL_CHILD, call_args([source], existing / "new", None))
    assert existing.is_dir() and list(existing.iterdir()) == []

    helpers.run(CALL_CHILD, call_args([source], existing, None))
    assert existing.is_dir() and list(existing.iterdir()) == []


def test_none_child_keeps_the_default_root(tmp_path: Path) -> None:
    """Keep the default root, shared with other processes, on exit."""
    source = write(tmp_path / "src" / "a.parquet", pl.DataFrame({"x": [1]}))
    helpers.run(CALL_CHILD, call_args([source], None, None))
    root = _disk.default_root()
    assert root.is_dir() and list(root.iterdir()) == []


def test_ttl_child_keeps_parts_on_exit(tmp_path: Path, sources: list[Path]) -> None:
    """Keep shared parts that have not expired after a normal exit."""
    root = tmp_path / "made" / "cache"
    helpers.run(CALL_CHILD, call_args(sources, root, 1))
    assert len(entries(root)) == 2
    assert len(list(root.rglob("*.parquet"))) == 6
    assert session_files(root) == []
    assert not list(root.rglob("lease-*"))


def test_zero_days_child_deletes_its_parts_on_exit(tmp_path: Path, sources: list[Path]) -> None:
    """Delete expired parts this process used when nobody else uses them."""
    root = tmp_path / "cache"
    helpers.run(CALL_CHILD, call_args(sources, root, 0))
    assert root.is_dir() and list(root.iterdir()) == []


def test_none_and_ttl_children_do_not_interfere(tmp_path: Path, sources: list[Path]) -> None:
    """Delete only the private parts when a process using both exits."""
    root = tmp_path / "cache"
    code = """
import json
import polars_helpers as plh
args = json.loads(sys.argv[1])
plh.scan_parquet_relaxed(args["paths"], directory=args["directory"], cache_ttl_days=None)
plh.scan_parquet_relaxed(args["paths"], directory=args["directory"], cache_ttl_days=1)
"""
    helpers.run(code, call_args(sources, root, None))
    assert not list(root.glob("*.parquet"))
    assert len(list(root.rglob("*.parquet"))) == 6


def test_killed_none_child_is_collected(tmp_path: Path, sources: list[Path]) -> None:
    """Delete the private parts of a killed process at the next call on its root."""
    root = tmp_path / "cache"
    elsewhere = tmp_path / "elsewhere"
    procs = [
        helpers.start(CALL_CHILD, call_args(sources, root, None)),
        helpers.start(CALL_CHILD, call_args(sources, elsewhere, None)),
    ]
    for proc in procs:
        helpers.ready(proc)
        helpers.kill(proc)
    assert len(list(root.glob("*.parquet"))) == 6
    assert len(session_files(root)) == 2
    other = write(tmp_path / "src2" / "c.parquet", pl.DataFrame({"x": [9]}))
    plh.scan_parquet_relaxed(other, directory=root, cache_ttl_days=None)
    assert len(list(root.glob("*.parquet"))) == 1
    assert len(session_files(root)) == 2  # only this process
    assert len(list(elsewhere.glob("*.parquet"))) == 6


# ------------------------------------------------------------ concurrency


def test_threads_share_complete_parts(tmp_path: Path, sources: list[Path]) -> None:
    """Prepare each part once when threads call at the same time."""
    root = tmp_path / "cache"
    results: list[object] = []
    errors: list[BaseException] = []

    def work(ttl: int | None) -> None:
        try:
            lf = plh.scan_parquet_relaxed(sources, directory=root / str(ttl), cache_ttl_days=ttl)
            results.append(lf.collect().to_dicts())
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(ttl,)) for ttl in [1, None, 0] * 4]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert results == [EXPECTED] * len(threads)
    assert len(generations(root / "1")) == 2
    assert len(list((root / "None").glob("*.parquet"))) == 6


def test_threads_cleaning_and_writing(tmp_path: Path, sources: list[Path]) -> None:
    """Never delete parts in use while other calls clean and write."""
    root = tmp_path / "cache"
    frames: list[pl.LazyFrame] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(3):
                frames.append(plh.scan_parquet_relaxed(sources, directory=root, cache_ttl_days=0))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert [f.collect().to_dicts() for f in frames] == [EXPECTED] * 12


def test_processes_share_complete_parts(tmp_path: Path, sources: list[Path]) -> None:
    """Prepare each part once when processes call at the same time."""
    root = tmp_path / "cache"
    procs = [helpers.start(CALL_CHILD, call_args(sources, root, 1)) for _ in range(3)]
    outputs = [json.loads(helpers.ready(p)) for p in procs]
    for proc in procs:
        helpers.finish(proc)
    assert outputs == [EXPECTED] * 3
    assert len(generations(root)) == 2
    assert len(list(root.rglob("*.parquet"))) == 6
    assert session_files(root) == []
