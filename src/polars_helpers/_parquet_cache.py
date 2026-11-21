"""Parts of ``scan_parquet_relaxed`` on disk: names, reuse, and lifetime.

A part is one column of one source file, cast to the common type of the
column, in a Parquet file of its own. This module decides where a part is
kept, reuses one that is still valid, publishes a new one only once it is
complete, and deletes parts that are no longer needed. What a part holds is
written by a function the caller passes.

With ``cache_ttl_days=None``, parts belong to the session of the process
(see ``_disk``) and are deleted when it exits. They are files directly in
the root, ``polars-helpers-parquet-<session>-<source>-<column>.parquet``.

With ``cache_ttl_days=n``, parts are shared by the processes of the user
and kept ``n`` days after their last use. Each source file has an entry,
``polars-helpers-parquet-<user>-ttl<n>-<source>/``, locked through
``<entry>.lock`` beside it, holding generations ``<entry>/<generation>/``.
A generation has a ``meta.json`` listing its complete parts and the time of
its last use, the parts ``<column>.parquet``, and one ``lease-<session>``
file per session that read it. A generation is deleted once it has expired
and no living session holds a lease on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from . import _disk

# Version of the layout and metadata of the parts; a change makes old parts
# unused.
_FORMAT = 1

_PREFIX = "polars-helpers-parquet-"
_MARKER = "polars-helpers-parquet"
_DAY_SECONDS = 86400
_GENERATION = re.compile(r"[0-9a-f]{12}")
_META = "meta.json"
_LEASE = "lease-"

# Attempts and their interval to lock an entry on exit, which does not
# wait for a long write of another process.
_EXIT_LOCK_ATTEMPTS = 50
_EXIT_LOCK_SECONDS = 0.02


def _now() -> float:
    """Give the current time.

    Returns
    -------
    float
        Seconds since the epoch, in UTC. Tests replace this function.
    """
    return time.time()


@dataclass(frozen=True)
class Source:
    """State of a source file when it was listed.

    Parameters
    ----------
    path : Path
        Absolute, resolved path.
    size : int
        Size in bytes.
    mtime_ns : int
        Modification time in nanoseconds.
    """

    path: Path
    size: int
    mtime_ns: int

    @classmethod
    def stat(cls, path: Path) -> Source:
        """Read the state of a file.

        Parameters
        ----------
        path : Path
            Absolute, resolved path.

        Returns
        -------
        Source
            Its current size and modification time.
        """
        st = os.stat(path)
        return cls(path, st.st_size, st.st_mtime_ns)


@dataclass(frozen=True)
class Column:
    """A part to prepare: one column and its type in the result.

    Parameters
    ----------
    name : str
        Column name.
    dtype : pl.DataType
        Type of the column in the result.
    """

    name: str
    dtype: pl.DataType


# Function writing the part of a column to a given path.
Writer = Callable[[Column, Path], None]

# Function raising if a source file changed since it was listed.
Check = Callable[[], None]


def _digest(*values: object) -> str:
    """Hash values to name files.

    Parameters
    ----------
    *values : object
        Values that ``json.dumps`` accepts.

    Returns
    -------
    str
        Hexadecimal SHA-256 of their JSON.
    """
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def _source_key(source: Source) -> str:
    """Name the parts of one state of a source file.

    Parameters
    ----------
    source : Source
        Source file.

    Returns
    -------
    str
        16 hexadecimal digits from the format, the polars version, the path,
        the size, and the modification time.
    """
    path = os.path.normcase(str(source.path))
    return _digest(_FORMAT, pl.__version__, path, source.size, source.mtime_ns)[:16]


def _column_key(column: Column) -> str:
    """Name the part of one column and type.

    Parameters
    ----------
    column : Column
        Column of the part.

    Returns
    -------
    str
        12 hexadecimal digits from the column name and the structure of its
        type.
    """
    return _digest(column.name, _describe(column.dtype))[:12]


def _describe(dtype: pl.DataType) -> str:
    """Describe a type to compare it with a type of a stored part.

    Parameters
    ----------
    dtype : pl.DataType
        Type.

    Returns
    -------
    str
        Its ``repr``, which shows its whole structure (inner types, fields,
        time units, time zones). It is compared as text and never evaluated.
    """
    return repr(dtype)


def prepare_parts(
    root: Path,
    ttl_days: int | None,
    source: Source,
    columns: list[Column],
    write: Writer,
    check: Check,
) -> list[Path]:
    """Give a complete part of each column of a source file.

    Parts that are still valid are reused; the missing ones are written to
    temporary names by ``write``, and published only after ``check``
    confirms that the source did not change. Parts of the same source are
    prepared by one call at a time, in this process and others.

    Parameters
    ----------
    root : Path
        Existing root of the parts.
    ttl_days : int | None
        Days to keep the parts after their last use, or ``None`` to keep
        them until the process exits.
    source : Source
        Source file, as listed.
    columns : list[Column]
        Columns to prepare.
    write : Writer
        Function writing the part of a column to a given path.
    check : Check
        Function raising if the source changed since it was listed.

    Returns
    -------
    list[Path]
        Part of each column, in the order of ``columns``. They are kept until
        the process exits.
    """
    session = _disk.get_session(root)
    if ttl_days is None:
        return _prepare_private(session, source, columns, write, check)
    return _prepare_shared(session, ttl_days, source, columns, write, check)


_private: set[str] = set()
_private_mutexes: dict[str, threading.Lock] = {}
_private_mutex = threading.Lock()


def _prepare_private(
    session: _disk.Session,
    source: Source,
    columns: list[Column],
    write: Writer,
    check: Check,
) -> list[Path]:
    """Prepare the parts of a source in the session of the process.

    Parameters
    ----------
    session : _disk.Session
        Session of this process in the root.
    source : Source
        Source file.
    columns : list[Column]
        Columns to prepare.
    write : Writer
        Function writing a part.
    check : Check
        Function raising if the source changed.

    Returns
    -------
    list[Path]
        Part of each column, in order.
    """
    base = f"{_PREFIX}{session.id}-{_source_key(source)}"
    with _private_mutex:
        mutex = _private_mutexes.setdefault(str(session.root / base), threading.Lock())
    with mutex:
        parts = [session.root / f"{base}-{_column_key(c)}.parquet" for c in columns]
        missing = [
            (c, part) for c, part in zip(columns, parts, strict=True) if str(part) not in _private or not part.exists()
        ]
        if missing:
            tmps = [part.with_name(f"{part.stem}.{uuid.uuid4().hex[:8]}.tmp") for _, part in missing]
            session.register([*tmps, *(part for _, part in missing)])
            _write_all(missing, tmps, write, check)
            for (_, part), tmp in zip(missing, tmps, strict=True):
                os.replace(tmp, part)
                _private.add(str(part))
        return parts


def _write_all(
    missing: list[tuple[Column, Path]],
    tmps: list[Path],
    write: Writer,
    check: Check,
) -> None:
    """Write parts to temporary files and check the source after.

    Parameters
    ----------
    missing : list[tuple[Column, Path]]
        Columns to write, with their final paths.
    tmps : list[Path]
        Temporary path of each.
    write : Writer
        Function writing a part.
    check : Check
        Function raising if the source changed.

    Raises
    ------
    BaseException
        What ``write`` or ``check`` raised, after the temporary files are
        deleted.
    """
    try:
        check()
        for (column, _), tmp in zip(missing, tmps, strict=True):
            write(column, tmp)
        check()
    except BaseException:
        _disk.delete_paths(tmps)
        raise


def _entry_name(ttl_days: int, source: Source) -> str:
    """Name the entry of a source file in a namespace.

    Parameters
    ----------
    ttl_days : int
        Days to keep the parts, the namespace.
    source : Source
        Source file.

    Returns
    -------
    str
        Name of the entry directory.
    """
    return f"{_PREFIX}{_disk.user_tag()}-ttl{ttl_days}-{_source_key(source)}"


def _is_valid(meta: dict[str, object] | None, now: float) -> bool:
    """Tell whether a generation is ours and complete and not expired.

    Parameters
    ----------
    meta : dict[str, object] | None
        Its metadata, or ``None`` if missing.
    now : float
        Current time.

    Returns
    -------
    bool
        Whether it can be reused.
    """
    return _is_ours(meta) and not _is_expired(meta, now)


def _is_ours(meta: dict[str, object] | None) -> bool:
    """Tell whether metadata was written by this package for this user.

    Parameters
    ----------
    meta : dict[str, object] | None
        Metadata, or ``None`` if missing.

    Returns
    -------
    bool
        Whether it has the marker, the format, and the user of this process.
    """
    return (
        meta is not None
        and meta.get("marker") == _MARKER
        and meta.get("format") == _FORMAT
        and meta.get("user") == _disk.user_id()
        and isinstance(meta.get("parts"), dict)
        and isinstance(meta.get("last_used_at"), (int, float))
        and isinstance(meta.get("ttl_days"), int)
    )


def _is_expired(meta: dict[str, object] | None, now: float) -> bool:
    """Tell whether a generation has expired.

    Parameters
    ----------
    meta : dict[str, object] | None
        Its metadata, as checked by ``_is_ours``.
    now : float
        Current time.

    Returns
    -------
    bool
        Whether ``now`` is at or after the last use plus the days to keep.
    """
    assert meta is not None
    last = meta["last_used_at"]
    days = meta["ttl_days"]
    assert isinstance(last, (int, float)) and isinstance(days, int)
    return now >= last + days * _DAY_SECONDS


def _leases(generation: Path) -> list[Path]:
    """List the lease files of a generation.

    Parameters
    ----------
    generation : Path
        Generation directory.

    Returns
    -------
    list[Path]
        Its lease files.
    """
    try:
        return [p for p in generation.iterdir() if p.name.startswith(_LEASE)]
    except OSError:
        return []


def _has_live_lease(root: Path, generation: Path) -> bool:
    """Tell whether a living session holds a lease on a generation.

    Lease files of ended sessions are deleted on the way.

    Parameters
    ----------
    root : Path
        Root of the sessions.
    generation : Path
        Generation directory.

    Returns
    -------
    bool
        Whether a lease of a living session remains.
    """
    live = False
    for lease in _leases(generation):
        if _disk.is_session_alive(root, lease.name[len(_LEASE) :]):
            live = True
        else:
            _disk.delete_paths([lease])
    return live


def _delete_generation(generation: Path) -> None:
    """Delete the files of a generation and then its directory.

    Only files this module writes are deleted; the directory is kept if
    anything else is in it or a file cannot be deleted.

    Parameters
    ----------
    generation : Path
        Generation directory.
    """
    try:
        files = [
            p
            for p in generation.iterdir()
            if p.name == _META
            or p.name.startswith(_LEASE)
            or p.suffix in (".parquet", ".tmp")
        ]
    except OSError:
        return
    # The metadata goes last, so that what is left is never taken as complete
    # parts of an incomplete generation.
    files.sort(key=lambda p: p.name == _META)
    _disk.delete_paths(files, [generation])


def _delete_incomplete(generation: Path, meta: dict[str, object]) -> None:
    """Delete the files of writes that did not complete in a generation.

    Parameters
    ----------
    generation : Path
        Generation directory.
    meta : dict[str, object]
        Its metadata, listing the complete parts.
    """
    parts = meta["parts"]
    assert isinstance(parts, dict)
    complete = {f"{key}.parquet" for key in parts}
    try:
        stray = [
            p
            for p in generation.iterdir()
            if p.suffix == ".tmp" or (p.suffix == ".parquet" and p.name not in complete)
        ]
    except OSError:
        return
    _disk.delete_paths(stray)


def _generations(entry: Path) -> list[Path]:
    """List the generation directories of an entry.

    Parameters
    ----------
    entry : Path
        Entry directory.

    Returns
    -------
    list[Path]
        Its generations; other files are ignored.
    """
    try:
        return [p for p in entry.iterdir() if p.is_dir() and _GENERATION.fullmatch(p.name)]
    except OSError:
        return []


def _prepare_shared(
    session: _disk.Session,
    ttl_days: int,
    source: Source,
    columns: list[Column],
    write: Writer,
    check: Check,
) -> list[Path]:
    """Prepare the parts of a source shared between processes.

    Parameters
    ----------
    session : _disk.Session
        Session of this process in the root.
    ttl_days : int
        Days to keep the parts after their last use.
    source : Source
        Source file.
    columns : list[Column]
        Columns to prepare.
    write : Writer
        Function writing a part.
    check : Check
        Function raising if the source changed.

    Returns
    -------
    list[Path]
        Part of each column, in order.
    """
    root = session.root
    entry = root / _entry_name(ttl_days, source)
    fd = _disk.lock_file(root / f"{entry.name}.lock", blocking=True)
    assert fd is not None
    try:
        entry.mkdir(exist_ok=True)
        now = _now()
        chosen: tuple[Path, dict[str, object]] | None = None
        if ttl_days > 0:
            for generation in _generations(entry):
                found = _disk.read_json(generation / _META)
                if found is None or not _is_valid(found, now) or found.get("ttl_days") != ttl_days:
                    continue
                if chosen is None or _last_used(found) > _last_used(chosen[1]):
                    chosen = (generation, found)
        if chosen is None:
            generation = entry / uuid.uuid4().hex[:12]
            generation.mkdir()
            meta: dict[str, object] = {
                "marker": _MARKER,
                "format": _FORMAT,
                "user": _disk.user_id(),
                "polars": pl.__version__,
                "source": {
                    "path": str(source.path),
                    "size": source.size,
                    "mtime_ns": source.mtime_ns,
                },
                "ttl_days": ttl_days,
                "created_at": None,
                "last_used_at": now,
                "parts": {},
            }
            new = True
        else:
            generation, meta = chosen
            new = False
        parts_meta = meta["parts"]
        assert isinstance(parts_meta, dict)
        keys = [_column_key(c) for c in columns]
        parts = [generation / f"{key}.parquet" for key in keys]
        missing = [
            (c, part)
            for c, key, part in zip(columns, keys, parts, strict=True)
            if not (
                isinstance(entry_meta := parts_meta.get(key), dict)
                and entry_meta.get("dtype") == _describe(c.dtype)
                and entry_meta.get("column") == c.name
            )
        ]
        tmps = [part.with_name(f"{part.stem}.{uuid.uuid4().hex[:8]}.tmp") for _, part in missing]
        try:
            _write_all(missing, tmps, write, check)
        except BaseException:
            if new:
                _delete_generation(generation)
            raise
        for (column, part), tmp in zip(missing, tmps, strict=True):
            os.replace(tmp, part)
            parts_meta[part.stem] = {
                "column": column.name,
                "dtype": _describe(column.dtype),
            }
        now = _now()
        if meta["created_at"] is None:
            meta["created_at"] = now
        meta["last_used_at"] = now
        lease = generation / f"{_LEASE}{session.id}"
        session.register([lease])
        lease.touch()
        _disk.write_json(generation / _META, meta)
        _add_lease(session, entry, generation)
        return parts
    finally:
        _disk.unlock_file(fd)


def _last_used(meta: dict[str, object]) -> float:
    """Give the time of the last use of a generation.

    Parameters
    ----------
    meta : dict[str, object]
        Its metadata, as checked by ``_is_ours``.

    Returns
    -------
    float
        Seconds since the epoch.
    """
    last = meta["last_used_at"]
    assert isinstance(last, (int, float))
    return float(last)


_held: dict[str, set[tuple[Path, Path]]] = {}
_held_mutex = threading.Lock()


def _add_lease(session: _disk.Session, entry: Path, generation: Path) -> None:
    """Remember a lease of the session to release it on exit.

    Parameters
    ----------
    session : _disk.Session
        Session holding the lease.
    entry : Path
        Entry directory.
    generation : Path
        Generation directory.
    """
    with _held_mutex:
        held = _held.get(session.id)
        if held is None:
            held = _held[session.id] = set()
            session.at_exit(lambda: _release_leases(session))
        held.add((entry, generation))


def _release_leases(session: _disk.Session) -> None:
    """Release the leases of a session and delete what nobody else needs.

    A generation that has expired and holds no other living lease is
    deleted, and so is its entry once empty. An entry that stays locked by
    another process is left for a later call.

    Parameters
    ----------
    session : _disk.Session
        Session that exits.
    """
    with _held_mutex:
        held = sorted(_held.pop(session.id, set()))
    for entry, generation in held:
        lease = generation / f"{_LEASE}{session.id}"
        fd = None
        for _ in range(_EXIT_LOCK_ATTEMPTS):
            fd = _disk.lock_file(session.root / f"{entry.name}.lock", blocking=False)
            if fd is not None:
                break
            time.sleep(_EXIT_LOCK_SECONDS)
        _disk.delete_paths([lease])
        if fd is None:
            continue
        try:
            _clean_generation(session.root, generation, _now())
            empty = _remove_if_empty(entry)
        finally:
            _disk.unlock_file(fd)
        if empty:
            _disk.delete_paths([session.root / f"{entry.name}.lock"])


def _clean_generation(root: Path, generation: Path, now: float) -> None:
    """Delete a generation, or its unfinished writes, if nobody needs them.

    Call it while holding the lock of the entry.

    Parameters
    ----------
    root : Path
        Root of the sessions.
    generation : Path
        Generation directory.
    now : float
        Current time.
    """
    meta = _disk.read_json(generation / _META)
    if meta is None:
        # Never completed: its writer held the lock and has ended.
        if not _leases(generation):
            _delete_generation(generation)
        return
    if not _is_ours(meta):
        return
    _delete_incomplete(generation, meta)
    if _is_expired(meta, now) and not _has_live_lease(root, generation):
        _delete_generation(generation)


def _remove_if_empty(entry: Path) -> bool:
    """Delete an entry directory if it holds no generation.

    Parameters
    ----------
    entry : Path
        Entry directory.

    Returns
    -------
    bool
        Whether the entry is gone.
    """
    try:
        entry.rmdir()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def collect_expired(root: Path) -> None:
    """Delete the expired shared parts of this user in a root.

    Entries locked by another process are skipped. Generations holding a
    lease of a living session are kept, even when expired.

    Parameters
    ----------
    root : Path
        Root of the parts.
    """
    prefix = f"{_PREFIX}{_disk.user_tag()}-ttl"
    try:
        names = {
            e.name.removesuffix(".lock") for e in os.scandir(root) if e.name.startswith(prefix)
        }
    except OSError:
        return
    now = _now()
    for name in sorted(names):
        entry = root / name
        lock_path = root / f"{name}.lock"
        fd = _disk.lock_file(lock_path, blocking=False)
        if fd is None:
            continue
        try:
            if entry.is_dir():
                for generation in _generations(entry):
                    _clean_generation(root, generation, now)
            empty = _remove_if_empty(entry) if entry.is_dir() or not entry.exists() else False
        finally:
            _disk.unlock_file(fd)
        if empty:
            _disk.delete_paths([lock_path])
