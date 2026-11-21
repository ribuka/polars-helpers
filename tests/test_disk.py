"""Tests for ``polars_helpers._disk``, shared by the helpers that write files."""

import ast
import json
import os
import sys
from pathlib import Path

import helpers
import polars as pl
import pytest

import polars_helpers as plh
from polars_helpers import _disk

SRC = Path(__file__).parents[1] / "src" / "polars_helpers"


SESSION_CHILD = """
from pathlib import Path
from polars_helpers import _disk
root = Path(sys.argv[1])
session = _disk.get_session(root)
part = root / f"part-{session.id}.bin"
session.register([part])
part.write_bytes(b"x")
(root / "sub").mkdir(exist_ok=True)
nested = root / "sub" / f"in-sub-{session.id}.bin"
session.register([nested])
nested.write_bytes(b"y")
print(session.id, flush=True)
wait_for_parent()
"""


# ------------------------------------------------------------ default root


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("{abs}", "{abs}/polars-helpers/cache"),
        (None, "{home}/AppData/Local/polars-helpers/cache"),
        ("", "{home}/AppData/Local/polars-helpers/cache"),
        ("relative/dir", "{home}/AppData/Local/polars-helpers/cache"),
    ],
)
def test_default_root_windows(tmp_path: Path, value: str | None, expected: str) -> None:
    """Use LOCALAPPDATA on Windows unless it is unset, empty, or relative."""
    home = tmp_path / "home"
    absolute = tmp_path / "local"
    environ = {} if value is None else {"LOCALAPPDATA": value.format(abs=absolute)}
    root = _disk._default_root_for("win32", environ, lambda: home)
    assert root == Path(expected.format(abs=absolute, home=home))


def test_default_root_macos(tmp_path: Path) -> None:
    """Use ~/Library/Caches on macOS, whatever XDG_CACHE_HOME says."""
    root = _disk._default_root_for(
        "darwin", {"XDG_CACHE_HOME": str(tmp_path / "xdg")}, lambda: tmp_path
    )
    assert root == tmp_path / "Library" / "Caches" / "polars-helpers"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("{abs}", "{abs}/polars-helpers"),
        (None, "{home}/.cache/polars-helpers"),
        ("", "{home}/.cache/polars-helpers"),
        ("relative/dir", "{home}/.cache/polars-helpers"),
    ],
)
def test_default_root_posix(tmp_path: Path, value: str | None, expected: str) -> None:
    """Use XDG_CACHE_HOME on other systems unless unset, empty, or relative."""
    home = tmp_path / "home"
    absolute = tmp_path / "xdg"
    environ = {} if value is None else {"XDG_CACHE_HOME": value.format(abs=absolute)}
    root = _disk._default_root_for("linux", environ, lambda: home)
    assert root == Path(expected.format(abs=absolute, home=home))


def test_default_root_is_redirected_in_tests(isolated_cache: Path) -> None:
    """Keep the default root of the tests out of the real user cache."""
    assert _disk.default_root().is_relative_to(isolated_cache)


@pytest.mark.parametrize(
    "module", ["_disk.py", "parquet.py", "_parquet_cache.py"]
)
def test_no_tempfile(module: str) -> None:
    """Never fall back on the temporary directory of the OS."""
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "tempfile" not in imported


# ------------------------------------------------------------- directory


def test_resolve_root_none_is_default() -> None:
    """Use the default root when no directory is given."""
    assert _disk.resolve_root(None) == _disk.default_root()


def test_resolve_root_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Take a relative directory from the working directory, as given."""
    monkeypatch.chdir(tmp_path)
    assert _disk.resolve_root("a/b") == tmp_path / "a" / "b"
    assert _disk.resolve_root(Path("~")) == tmp_path / "~"
    assert _disk.resolve_root("%X%") == tmp_path / "%X%"
    assert not (tmp_path / "a").exists()


@pytest.mark.parametrize(
    ("directory", "error"),
    [(1, TypeError), (b"dir", TypeError), ("", ValueError)],
)
def test_resolve_root_invalid(directory: object, error: type[Exception]) -> None:
    """Reject a directory of a wrong type or an empty string."""
    with pytest.raises(error, match="directory"):
        _disk.resolve_root(directory)  # type: ignore[arg-type]


def test_resolve_root_file(tmp_path: Path) -> None:
    """Reject an existing file."""
    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(NotADirectoryError):
        _disk.resolve_root(file)


def test_make_directory(tmp_path: Path) -> None:
    """Create a directory with its parents and list what was created."""
    target = tmp_path / "a" / "b" / "c"
    assert _disk.make_directory(target) == [target, target.parent, target.parent.parent]
    assert target.is_dir()
    assert _disk.make_directory(target) == []


def test_make_directory_error_is_raised(tmp_path: Path) -> None:
    """Raise the error of the OS when the directory cannot be made."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    with pytest.raises(OSError):
        _disk.make_directory(blocker / "sub")


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions"
)
def test_unwritable_root_raises_permission_error(tmp_path: Path) -> None:
    """Raise PermissionError when the root cannot be written, without fallback."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            _disk.make_directory(locked / "root")
        with pytest.raises(PermissionError):
            _disk.get_session(locked)
    finally:
        locked.chmod(0o700)


def test_delete_paths_skips_what_it_cannot_delete(tmp_path: Path) -> None:
    """Delete files and empty directories, skipping the others."""
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep").write_text("x")
    empty = tmp_path / "empty"
    empty.mkdir()
    file = tmp_path / "file"
    file.write_text("x")
    _disk.delete_paths([file, tmp_path / "missing"], [empty, full])
    assert not file.exists() and not empty.exists()
    assert (full / "keep").exists()


# ----------------------------------------------------------------- locks


def test_lock_file_is_exclusive(tmp_path: Path) -> None:
    """Refuse a second lock until the first is released."""
    path = tmp_path / "x.lock"
    fd = _disk.lock_file(path, blocking=True)
    assert fd is not None
    assert _disk.lock_file(path, blocking=False) is None
    _disk.unlock_file(fd)
    fd = _disk.lock_file(path, blocking=False)
    assert fd is not None
    _disk.unlock_file(fd)
    assert _disk.lock_file(tmp_path / "missing.lock", blocking=False, create=False) is None


def test_lock_is_released_when_the_process_is_killed(tmp_path: Path) -> None:
    """Free the lock of a killed process."""
    path = tmp_path / "x.lock"
    proc = helpers.start(
        "from pathlib import Path\nfrom polars_helpers import _disk\n"
        "fd = _disk.lock_file(Path(sys.argv[1]), blocking=True)\n"
        "print('ready', flush=True)\nwait_for_parent()\n",
        str(path),
    )
    try:
        helpers.ready(proc)
        assert _disk.lock_file(path, blocking=False) is None
    finally:
        helpers.kill(proc)
    fd = _disk.lock_file(path, blocking=False)
    assert fd is not None
    _disk.unlock_file(fd)


# -------------------------------------------------------------- sessions


def session_files(root: Path) -> list[str]:
    """List the session files directly in a root.

    Parameters
    ----------
    root : Path
        Root.

    Returns
    -------
    list[str]
        Their names, sorted.
    """
    return sorted(p.name for p in root.iterdir() if p.name.startswith(_disk.SESSION_PREFIX))


def test_session_files_are_in_the_root(tmp_path: Path) -> None:
    """Keep a lock and a JSON file named after the session in the root."""
    root = tmp_path / "root"
    root.mkdir()
    session = _disk.get_session(root)
    assert _disk.get_session(root) is session
    assert session_files(root) == [
        f".polars-helpers-session-{session.id}.json",
        f".polars-helpers-session-{session.id}.lock",
    ]
    data = json.loads(session.json_path.read_text(encoding="utf-8"))
    assert data["user"] == _disk.user_id()
    assert data["marker"] == "polars-helpers-session"
    assert _disk.is_session_alive(root, session.id)


def test_session_close_deletes_what_it_registered(tmp_path: Path) -> None:
    """Delete the registered files, the session files, and empty directories."""
    root = tmp_path / "root"
    created = _disk.make_directory(root / "a" / "b")
    session = _disk.Session(root / "a" / "b")
    part = root / "a" / "b" / "part"
    session.register([part])
    part.write_text("x")
    other = root / "a" / "b" / "other"
    other.write_text("x")
    session.register_directories(created)
    calls: list[str] = []
    session.at_exit(lambda: calls.append("called"))
    session.close()
    assert calls == ["called"]
    assert not part.exists()
    assert other.exists()
    assert session_files(root / "a" / "b") == []
    other.unlink()
    session2 = _disk.Session(root / "a" / "b")
    session2.register_directories(created)
    session2.close()
    assert not root.exists()


def child_session_script(directory: Path, *, ttl_none_dir: bool) -> str:
    """Build a child script that creates a session and exits normally.

    Parameters
    ----------
    directory : Path
        Directory to create and use as the root.
    ttl_none_dir : bool
        Whether to register the created directories to delete on exit.

    Returns
    -------
    str
        The script.
    """
    return (
        "from pathlib import Path\nfrom polars_helpers import _disk\n"
        f"root = _disk.resolve_root({str(directory)!r})\n"
        "created = _disk.make_directory(root)\n"
        "s = _disk.get_session(root)\n"
        + ("s.register_directories(created)\n" if ttl_none_dir else "")
        + "p = root / 'part'\ns.register([p])\np.write_text('x')\n"
        "import json\nprint(json.dumps(sorted(q.name for q in root.iterdir())))\n"
    )


def test_child_exit_leaves_nothing(tmp_path: Path) -> None:
    """Leave no session file, and delete created directories once empty."""
    existing = tmp_path / "existing"
    existing.mkdir()
    target = existing / "new" / "root"
    out = helpers.run(child_session_script(target, ttl_none_dir=True))
    names = json.loads(out)
    assert "part" in names
    assert any(n.startswith(".polars-helpers-session-") and n.endswith(".lock") for n in names)
    assert existing.is_dir()
    assert list(existing.iterdir()) == []

    kept = tmp_path / "kept"
    helpers.run(child_session_script(kept, ttl_none_dir=False))
    assert kept.is_dir() and list(kept.iterdir()) == []


def test_killed_session_is_collected(tmp_path: Path) -> None:
    """Delete the registered files and session files of a killed session."""
    root = tmp_path / "root"
    root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()
    unrelated = root / "unrelated.txt"
    unrelated.write_text("x")
    procs = {}
    for name, r in (("alive", root), ("killed", root), ("elsewhere", other_root)):
        procs[name] = helpers.start(SESSION_CHILD, str(r))
    try:
        ids = {name: helpers.ready(proc) for name, proc in procs.items()}
        helpers.kill(procs["killed"])
        helpers.kill(procs["elsewhere"])
        assert not _disk.is_session_alive(root, ids["killed"])
        assert _disk.is_session_alive(root, ids["alive"])

        _disk.collect_stale_sessions(root)
        names = session_files(root)
        assert f".polars-helpers-session-{ids['killed']}.lock" not in names
        assert f".polars-helpers-session-{ids['killed']}.json" not in names
        assert f".polars-helpers-session-{ids['alive']}.lock" in names
        assert f".polars-helpers-session-{ids['alive']}.json" in names
        assert not (root / f"part-{ids['killed']}.bin").exists()
        assert not (root / "sub" / f"in-sub-{ids['killed']}.bin").exists()
        assert (root / f"part-{ids['alive']}.bin").exists()
        assert (root / "sub").is_dir()
        assert (other_root / f"part-{ids['elsewhere']}.bin").exists()
        assert unrelated.exists()
        assert len(session_files(other_root)) == 2
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                assert proc.stdin is not None
                proc.stdin.write("\n")
                proc.stdin.flush()
                proc.communicate(timeout=60)
    assert session_files(root) == []
    assert not (root / f"part-{ids['alive']}.bin").exists()


def test_killed_session_files_are_deleted(tmp_path: Path) -> None:
    """Delete the registered files of a killed session only."""
    root = tmp_path / "root"
    root.mkdir()
    proc = helpers.start(SESSION_CHILD, str(root))
    try:
        sid = helpers.ready(proc)
    finally:
        helpers.kill(proc)
    assert (root / f"part-{sid}.bin").exists()
    _disk.collect_stale_sessions(root)
    assert not (root / f"part-{sid}.bin").exists()
    assert not (root / "sub" / f"in-sub-{sid}.bin").exists()
    assert (root / "sub").is_dir()
    assert session_files(root) == []


def test_sessions_of_other_users_are_kept(tmp_path: Path) -> None:
    """Never collect a session whose JSON file names another user."""
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "theirs.bin"
    victim.write_text("x")
    lock = root / ".polars-helpers-session-0123456789abcdef.lock"
    lock.write_text("")
    data = root / ".polars-helpers-session-0123456789abcdef.json"
    data.write_text(
        json.dumps({"marker": "polars-helpers-session", "user": "someone-else", "files": [str(victim)]})
    )
    _disk.collect_stale_sessions(root)
    assert victim.exists() and lock.exists() and data.exists()


def test_collect_never_leaves_the_root(tmp_path: Path) -> None:
    """Skip registered files outside the root of a session."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_text("x")
    (root / ".polars-helpers-session-0123456789abcdef.lock").write_text("")
    (root / ".polars-helpers-session-0123456789abcdef.json").write_text(
        json.dumps(
            {"marker": "polars-helpers-session", "user": _disk.user_id(), "files": [str(outside)]}
        )
    )
    _disk.collect_stale_sessions(root)
    assert outside.exists()
    assert session_files(root) == []


def test_unusable_default_root_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise the error of the OS instead of falling back on another directory."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    for name in ("LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(blocker))
    monkeypatch.setenv("HOME", str(blocker))
    monkeypatch.setenv("USERPROFILE", str(blocker))
    source = tmp_path / "a.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(source)
    with pytest.raises(OSError):
        plh.scan_parquet_relaxed(source)
    assert blocker.read_text() == "x"
