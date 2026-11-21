"""Stack Parquet files whose columns and types differ, with little memory."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from . import _disk, _parquet_cache
from ._parquet_cache import Column, Source
from .utils import natural_keys


def _scan(path: Path) -> pl.LazyFrame:
    """Scan one Parquet file as it is.

    Parameters
    ----------
    path : Path
        File to scan.

    Returns
    -------
    pl.LazyFrame
        Frame reading the file, without hive partitions or glob expansion.
    """
    return pl.scan_parquet(path, hive_partitioning=False, glob=False)


def _check_columns(columns: Sequence[str] | None) -> list[str] | None:
    """Check the ``columns`` argument.

    Parameters
    ----------
    columns : Sequence[str] | None
        Columns given by the caller.

    Returns
    -------
    list[str] | None
        The columns as a list, or ``None`` for every column.

    Raises
    ------
    TypeError
        If ``columns`` is a single ``str``, not a sequence (a generator, for
        example), or has an item that is not a ``str``.
    ValueError
        If ``columns`` is empty or names a column twice.
    """
    if columns is None:
        return None
    if isinstance(columns, str) or not isinstance(columns, Sequence):
        raise TypeError(f"columns must be a sequence of str or None, got {type(columns)}")
    names = list(columns)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"columns must hold only str, got {type(name)}")
    if not names:
        raise ValueError("columns must not be empty")
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"columns names a column more than once: {duplicated}")
    return names


def _check_ttl(cache_ttl_days: int | None) -> None:
    """Check the ``cache_ttl_days`` argument.

    Parameters
    ----------
    cache_ttl_days : int | None
        Days given by the caller.

    Raises
    ------
    TypeError
        If it is not an ``int`` or ``None``; a ``bool`` is not an ``int``
        here.
    ValueError
        If it is negative.
    """
    if cache_ttl_days is None:
        return
    if isinstance(cache_ttl_days, bool) or not isinstance(cache_ttl_days, int):
        raise TypeError(f"cache_ttl_days must be an int or None, got {type(cache_ttl_days)}")
    if cache_ttl_days < 0:
        raise ValueError(f"cache_ttl_days must not be negative, got {cache_ttl_days}")


def _same_path_key(path: Path) -> str:
    """Normalize a resolved path to find duplicates.

    Parameters
    ----------
    path : Path
        Resolved path.

    Returns
    -------
    str
        The path in normal case, so that paths differing only in case are
        the same on Windows.
    """
    return os.path.normcase(str(path))


def _dedupe(paths: list[Path]) -> list[Path]:
    """Drop repeated paths, keeping the first of each.

    Parameters
    ----------
    paths : list[Path]
        Resolved paths.

    Returns
    -------
    list[Path]
        Paths without duplicates, in their first order.
    """
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = _same_path_key(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _list_folder(folder: Path, recursive: bool) -> list[Path]:
    """List the Parquet files in a folder.

    Parameters
    ----------
    folder : Path
        Resolved folder.
    recursive : bool
        Whether to look in subfolders too; symbolic links and junctions to
        folders are not followed.

    Returns
    -------
    list[Path]
        Files whose extension is ``.parquet`` in any case. Files and folders
        whose name starts with ``.`` or ``_`` are skipped.
    """
    out: list[Path] = []
    with os.scandir(folder) as entries:
        children = list(entries)
    for entry in children:
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_dir(follow_symlinks=False) and not entry.is_junction():
            if recursive:
                out.extend(_list_folder(Path(entry.path), recursive))
        elif entry.is_file() and entry.name.lower().endswith(".parquet"):
            out.append(Path(entry.path))
    return out


def _sort_key(path: Path, base: Path | None) -> tuple[list[list[int | str]], str]:
    """Build the key that puts the files in natural order.

    Parameters
    ----------
    path : Path
        Resolved file.
    base : Path | None
        Path the files are compared from, or ``None`` to compare absolute
        paths.

    Returns
    -------
    tuple[list[list[int | str]], str]
        The natural key of each component of the relative path, compared
        without case, then the path itself to break ties.
    """
    relative = Path(os.path.relpath(path, base)) if base is not None else path
    return [natural_keys(part.casefold()) for part in relative.parts], path.as_posix()


def _list_files(paths: str | Path | Sequence[str | Path], recursive: bool) -> list[Path]:
    """List the source files in the order they are stacked.

    Parameters
    ----------
    paths : str | Path | Sequence[str | Path]
        A file, a folder, or a sequence of files.
    recursive : bool
        Whether to look in the subfolders of a folder.

    Returns
    -------
    list[Path]
        Resolved files without duplicates, in natural order.

    Raises
    ------
    TypeError
        If ``paths`` or an item of it is not a ``str`` or ``Path``.
    ValueError
        If ``paths`` is empty, an empty string, mixes a folder with other
        paths, or is a folder without Parquet files.
    FileNotFoundError
        If a path does not exist.
    """
    if isinstance(paths, (str, Path)):
        items: list[object] = [paths]
    elif isinstance(paths, Sequence) and not isinstance(paths, (bytes, bytearray)):
        items = list(paths)
    else:
        raise TypeError(f"paths must be a str, Path, or a sequence of them, got {type(paths)}")
    if not items:
        raise ValueError("paths must not be empty")
    resolved: list[Path] = []
    for item in items:
        if not isinstance(item, (str, Path)):
            raise TypeError(f"paths must hold only str or Path, got {type(item)}")
        if isinstance(item, str) and not item:
            raise ValueError("paths must not hold an empty string")
        path = Path(item).resolve()
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: {item}")
        resolved.append(path)
    resolved = _dedupe(resolved)

    if len(resolved) == 1 and resolved[0].is_dir():
        base: Path | None = resolved[0]
        files = _dedupe([f.resolve() for f in _list_folder(resolved[0], recursive)])
        if not files:
            raise ValueError(f"no Parquet file in the folder: {resolved[0]}")
    else:
        folders = [str(p) for p in resolved if p.is_dir()]
        if folders:
            raise ValueError(f"a folder must be the only path, got folders {folders} among {len(resolved)} paths")
        files = resolved
        try:
            base = Path(os.path.commonpath([f.parent for f in files]))
        except ValueError:
            base = None
    return sorted(files, key=lambda f: _sort_key(f, base))


def _common_dtype(name: str, found: list[tuple[Path, pl.DataType]]) -> pl.DataType:
    """Find the common type of a column like ``vertical_relaxed`` does.

    Parameters
    ----------
    name : str
        Column name.
    found : list[tuple[Path, pl.DataType]]
        Type of the column in each file holding it, in file order.

    Returns
    -------
    pl.DataType
        Type that ``pl.concat(how="vertical_relaxed")`` gives the column.

    Raises
    ------
    ValueError
        If the types have no common type; the polars exception is its cause.
    """
    dtype = found[0][1]
    for i, (path, other) in enumerate(found[1:], start=1):
        try:
            dtype = pl.concat(
                [pl.DataFrame(schema={name: dtype}), pl.DataFrame(schema={name: other})],
                how="vertical_relaxed",
            ).schema[name]
        except pl.exceptions.PolarsError as e:
            files = ", ".join(str(p) for p, _ in found[:i])
            raise ValueError(
                f"column {name!r} has no common type: {dtype} in {files} and {other} in {path}"
            ) from e
    return dtype


def _target_columns(
    files: list[Path], schemas: list[pl.Schema], columns: list[str] | None
) -> list[Column]:
    """Choose the columns of the result and their types.

    Parameters
    ----------
    files : list[Path]
        Source files, in order.
    schemas : list[pl.Schema]
        Schema of each file.
    columns : list[str] | None
        Columns to keep, or ``None`` for every column.

    Returns
    -------
    list[Column]
        Columns in the order of ``columns``, or in the order they first
        appear, with their common types.

    Raises
    ------
    pl.exceptions.ColumnNotFoundError
        If a column of ``columns`` is in no file.
    ValueError
        If the files have no column, or a column has no common type.
    """
    seen: dict[str, None] = {}
    for schema in schemas:
        seen.update(dict.fromkeys(schema.names()))
    if columns is None:
        if not seen:
            raise ValueError(f"the files have no column: {[str(f) for f in files]}")
        names = list(seen)
    else:
        unknown = [name for name in columns if name not in seen]
        if unknown:
            raise pl.exceptions.ColumnNotFoundError(
                f"columns not in any file: {unknown}; the files have {list(seen)}"
            )
        names = columns
    return [
        Column(
            name,
            _common_dtype(
                name,
                [(f, s[name]) for f, s in zip(files, schemas, strict=True) if name in s],
            ),
        )
        for name in names
    ]


def _part_writer(path: Path, schema: pl.Schema) -> _parquet_cache.Writer:
    """Make the function writing the parts of one source file.

    Parameters
    ----------
    path : Path
        Source file.
    schema : pl.Schema
        Its schema.

    Returns
    -------
    _parquet_cache.Writer
        Function writing the part of a column: the column cast to its common
        type with ``strict=True``, or, for a column the file lacks, nulls of
        that type, one per row of the file.
    """
    rows: list[int] = []

    def write(column: Column, out: Path) -> None:
        if column.name in schema:
            source_dtype: object = schema[column.name]
            frame = _scan(path).select(pl.col(column.name).cast(column.dtype, strict=True))
        else:
            source_dtype = "missing"
            if not rows:
                rows.append(_scan(path).select(pl.len()).collect().item())
            frame = pl.LazyFrame().select(
                pl.repeat(None, rows[0], dtype=column.dtype).alias(column.name)
            )
        try:
            frame.sink_parquet(out)
        except pl.exceptions.PolarsError as e:
            e.add_note(
                f"while writing column {column.name!r} of {path} "
                f"from {source_dtype} to {column.dtype}"
            )
            raise

    return write


def _source_check(source: Source) -> _parquet_cache.Check:
    """Make the function checking that a source file did not change.

    Parameters
    ----------
    source : Source
        Source file, as listed.

    Returns
    -------
    _parquet_cache.Check
        Function raising ``RuntimeError`` if the size or the modification
        time of the file changed.
    """

    def check() -> None:
        if Source.stat(source.path) != source:
            raise RuntimeError(f"the file changed while its parts were written: {source.path}")

    return check


def scan_parquet_relaxed(
    paths: str | Path | Sequence[str | Path],
    *,
    recursive: bool = False,
    columns: Sequence[str] | None = None,
    directory: str | Path | None = None,
    cache_ttl_days: int | None = 1,
) -> pl.LazyFrame:
    r"""Stack Parquet files whose columns and types differ, with little memory.

    The result has the rows of ``pl.concat(..., how="diagonal_relaxed")`` of
    the files: their columns in the order they first appear, each with the
    type that ``vertical_relaxed`` gives it, and nulls where a file lacks a
    column. It is built on disk without loading the files into memory: each
    column of each file is cast to its common type with ``strict=True`` and
    written, one at a time with the streaming engine, to a Parquet file of
    its own, called a part. The parts of a column are read by one
    ``pl.scan_parquet``, and the columns are put side by side.

    The parts are prepared inside this call. A ``select`` or ``filter`` on
    the returned frame applies when the parts are read; it does not reduce
    the work of this call. With ``columns=None`` the first call writes every
    column of every file; pass ``columns`` to prepare only some. Collecting
    the whole result loads it into memory; aggregate it or ``sink_*`` it
    when it is large.

    Parameters
    ----------
    paths : str | Path | Sequence[str | Path]
        A Parquet file, a folder, or a sequence of Parquet files. In a
        folder, the files whose extension is ``.parquet`` in any case are
        read, except those whose name, or the name of a folder on the way,
        starts with ``.`` or ``_``. Paths are resolved and read once each.
        Files are stacked in the natural order of their paths relative to
        the folder, or to the common parent of the files, compared
        component by component without case (``data_2`` before
        ``data_10``); the rows of each file keep their order.
    recursive : bool, default False
        Whether to look in the subfolders of a folder too; symbolic links to
        folders are not followed. It does not affect files.
    columns : Sequence[str] | None, default None
        Columns of the result, in this order; only they are written. A
        column some files lack is null there. The types of the other columns
        are not looked at. ``None`` keeps every column.
    directory : str | Path | None, default None
        Directory to keep the parts in; it is created with its parents if
        missing, and a relative path is taken from the current working
        directory. Files already in it are neither read, overwritten, nor
        deleted unless they are parts and session files this package wrote
        for the same user. If ``None``, the user cache directory of the OS
        is used: ``%LOCALAPPDATA%\polars-helpers\cache`` on Windows,
        ``~/Library/Caches/polars-helpers`` on macOS, and
        ``$XDG_CACHE_HOME/polars-helpers`` (or ``~/.cache/polars-helpers``)
        elsewhere. The temporary directory of the OS is never used.
    cache_ttl_days : int | None, default 1
        Days to keep the parts after their last use by this function, so
        that later calls, in this process or another, reuse them while the
        files keep their size and modification time. The time limit is the
        last use plus ``cache_ttl_days`` times 24 hours; reusing the parts
        extends it, collecting the returned frame does not. ``0`` keeps
        nothing for later calls. ``None`` keeps the parts only for this
        process: they are reused by later calls in it and deleted when it
        exits normally, together with the directories this call created for
        ``directory`` once they are empty.

    Returns
    -------
    pl.LazyFrame
        Frame reading the parts. It reads the files as they were during
        this call, even after they change, and its parts are kept until the
        process exits, even past their time limit.

    Raises
    ------
    TypeError
        If an argument has a wrong type: ``paths`` not a path or a sequence
        of paths, ``recursive`` not a ``bool``, ``columns`` a single ``str``
        or not a sequence of ``str``, ``directory`` not a path or ``None``,
        or ``cache_ttl_days`` not an ``int`` or ``None``.
    ValueError
        If ``paths`` is empty or an empty string, mixes a folder with other
        paths, or is a folder without Parquet files; if ``columns`` is empty
        or repeats a name; if ``directory`` is an empty string; if
        ``cache_ttl_days`` is negative; if the files have no column; or if a
        column has no common type, for example datetimes of different time
        zones.
    FileNotFoundError
        If a path does not exist.
    NotADirectoryError
        If ``directory`` is an existing file.
    pl.exceptions.ColumnNotFoundError
        If a column of ``columns`` is in no file.
    RuntimeError
        If a file changes while its parts are written.

    Notes
    -----
    Every check is done before any part is written. A polars error while
    writing a part keeps its type and gets a note naming the file, the
    column, and both types; I/O errors propagate as they are. The
    unfinished parts of a failed call are deleted before the exception is
    raised.

    Each call also deletes, in the directory it uses, the parts of ended
    processes kept with ``None`` and the parts whose time limit has passed,
    unless a living process still reads them. Nothing deletes them between
    calls, and parts are never deleted to save space, so the directory can
    grow; delete it by hand when no process uses it.
    """
    if not isinstance(recursive, bool):
        raise TypeError(f"recursive must be a bool, got {type(recursive)}")
    names = _check_columns(columns)
    _check_ttl(cache_ttl_days)
    root = _disk.resolve_root(directory)
    files = _list_files(paths, recursive)

    sources = [Source.stat(f) for f in files]
    schemas = [_scan(f).collect_schema() for f in files]
    targets = _target_columns(files, schemas, names)

    created = _disk.make_directory(root)
    session = _disk.get_session(root)
    if cache_ttl_days is None and directory is not None:
        session.register_directories(created)
    _disk.collect_stale_sessions(root)
    _parquet_cache.collect_expired(root)

    parts: list[list[Path]] = [[] for _ in targets]
    for source, schema in zip(sources, schemas, strict=True):
        for column_parts, part in zip(
            parts,
            _parquet_cache.prepare_parts(
                root,
                cache_ttl_days,
                source,
                targets,
                _part_writer(source.path, schema),
                _source_check(source),
            ),
            strict=True,
        ):
            column_parts.append(part)

    scans = [pl.scan_parquet(p, hive_partitioning=False, glob=False) for p in parts]
    if len(scans) == 1:
        return scans[0]
    return pl.concat(scans, how="horizontal")
