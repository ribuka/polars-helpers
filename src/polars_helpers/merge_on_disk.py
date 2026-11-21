"""Merge two large polars LazyFrames with little memory by working on disk."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import polars as pl

from . import _disk
from .merge import HOW, _print_summary, _resolve_keys, merge

# Base names of the temporary row numbers of each side and of the partition
# key holding the bucket of each row; ``_temporary_names`` adds a suffix when
# a base name is taken by a column of the merge.
_LEFT_ROW = "__polars_helpers_left_row"
_RIGHT_ROW = "__polars_helpers_right_row"
_BUCKET = "__polars_helpers_bucket"

# Fixed seed of the key hash, so both sides put equal keys in one bucket.
_HASH_SEED = 0

# Target ``estimated_size()`` of the merged rows put back in order at a time;
# sorting them makes the peak about twice as large.
_RESTORE_BYTES = 1 << 30

# Most merged rows put back in order at a time, however narrow they are.
_RESTORE_ROWS = 10_000_000

# Merged rows read from the start to measure the size of one row.
_SAMPLE_ROWS = 10_000

# Width of the intervals of row numbers whose merged rows are counted first;
# an interval holding more rows than fit at a time is counted again row by
# row number.
_COUNT_BLOCK = 100_000

# Rows per row group of the merged buckets, so that putting one range of
# row numbers back in order reads only the row groups of that range.
_ROW_GROUP_ROWS = 1_000_000


def _temporary_names(taken: set[str], bases: list[str]) -> list[str]:
    """Choose temporary column names that no column of the merge uses.

    Parameters
    ----------
    taken : set[str]
        Names to avoid: the columns of both sides, the names a join may give
        them with the ``"_right"`` suffix, and the indicator.
    bases : list[str]
        Preferred names, one per temporary column.

    Returns
    -------
    list[str]
        One name per base, distinct from ``taken`` and from each other.
    """
    names: list[str] = []
    avoid = set(taken)
    for base in bases:
        name, n = base, 0
        while name in avoid:
            name, n = f"{base}_{n}", n + 1
        avoid.add(name)
        names.append(name)
    return names


def _bucket_expr(
    keys: list[str], dtypes: list[pl.DataType], buckets: int, name: str
) -> pl.Expr:
    """Build the bucket of each row from the hash of its keys.

    Parameters
    ----------
    keys : list[str]
        Key columns of one side.
    dtypes : list[pl.DataType]
        Common type of each key pair of both sides, so that equal keys of
        both sides hash alike.
    buckets : int
        Number of buckets.
    name : str
        Name of the bucket column.

    Returns
    -------
    pl.Expr
        Bucket number from ``0`` to ``buckets - 1``. The key columns
        themselves are not changed.
    """
    fields = [
        pl.col(key).cast(dtype, strict=False).alias(str(i))
        for i, (key, dtype) in enumerate(zip(keys, dtypes, strict=True))
    ]
    return (pl.struct(fields).hash(seed=_HASH_SEED) % buckets).alias(name)


def _common_key_dtypes(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    left_keys: list[str],
    right_keys: list[str],
) -> list[pl.DataType]:
    """Find the common type of each key pair of both sides.

    Parameters
    ----------
    left, right : pl.LazyFrame
        Frames to merge.
    left_keys, right_keys : list[str]
        Key columns of each side.

    Returns
    -------
    list[pl.DataType]
        Supertype of each key pair, as ``pl.concat(how="vertical_relaxed")``
        finds it.
    """
    names = [str(i) for i in range(len(left_keys))]
    schema = pl.concat(
        [
            left.clear().select(pl.col(k).alias(n) for k, n in zip(left_keys, names, strict=True)),
            right.clear().select(pl.col(k).alias(n) for k, n in zip(right_keys, names, strict=True)),
        ],
        how="vertical_relaxed",
    ).collect_schema()
    return [schema[n] for n in names]


def _align_full_keys(
    frame: pl.LazyFrame, keys: list[str], dtypes: list[pl.DataType], how: HOW
) -> pl.LazyFrame:
    """Cast the keys of one side to the common types for a full join.

    A full join of keys of different types, such as Int32 and Int64, gives
    the common type when collected but declares the left type in its
    schema, and the streaming engine fails to write it. Casting both sides
    to the common type first gives the type that ``merge`` collects.

    Parameters
    ----------
    frame : pl.LazyFrame
        One side of the merge.
    keys : list[str]
        Key columns of that side.
    dtypes : list[pl.DataType]
        Common type of each key pair of both sides.
    how : HOW
        Join strategy of the merge.

    Returns
    -------
    pl.LazyFrame
        ``frame``, with its keys cast when ``how`` is ``"full"`` and their
        types differ from the common types; otherwise unchanged.
    """
    if how != "full":
        return frame
    schema = frame.collect_schema()
    casts = [
        pl.col(key).cast(dtype)
        for key, dtype in zip(keys, dtypes, strict=True)
        if schema[key] != dtype
    ]
    return frame.with_columns(casts) if casts else frame


def _partition(frame: pl.LazyFrame, bucket: pl.Expr, directory: Path) -> dict[int, list[Path]]:
    """Write a frame split into buckets and list the files of each bucket.

    Parameters
    ----------
    frame : pl.LazyFrame
        Frame to split.
    bucket : pl.Expr
        Bucket of each row, from ``_bucket_expr``.
    directory : Path
        Directory to write the buckets under.

    Returns
    -------
    dict[int, list[Path]]
        Files of each bucket that holds rows, in the order they were written.
    """
    frame.sink_parquet(
        pl.PartitionBy(directory, key=bucket, include_key=False),
        engine="streaming",
        mkdir=True,
    )
    files: dict[int, list[Path]] = {}
    if not directory.is_dir():
        return files
    for sub in directory.iterdir():
        if sub.is_dir():
            files[int(sub.name.rpartition("=")[2])] = sorted(sub.glob("*.parquet"))
    return files


def _scan(files: list[Path], schema: pl.Schema) -> pl.LazyFrame:
    """Read the files of one bucket, or an empty frame if there are none.

    Parameters
    ----------
    files : list[Path]
        Files of the bucket.
    schema : pl.Schema
        Schema of the frame the files were written from.

    Returns
    -------
    pl.LazyFrame
        Rows of the bucket.
    """
    if not files:
        return pl.LazyFrame(schema=schema)
    return pl.scan_parquet(files)


def _restore_limit(merged: pl.LazyFrame) -> int:
    """Choose how many merged rows to put back in order at a time.

    Parameters
    ----------
    merged : pl.LazyFrame
        Merged rows of every bucket, with the row-number columns.

    Returns
    -------
    int
        ``_RESTORE_BYTES`` divided by the size of one row, measured with
        ``estimated_size()`` on the first ``_SAMPLE_ROWS`` rows; at most
        ``_RESTORE_ROWS`` and at least ``1``. ``_RESTORE_ROWS`` if there are
        no rows.
    """
    sample = merged.head(_SAMPLE_ROWS).collect()
    if sample.height == 0:
        return _RESTORE_ROWS
    width = sample.estimated_size() / sample.height
    if width <= 0:
        return _RESTORE_ROWS
    return max(1, min(_RESTORE_ROWS, int(_RESTORE_BYTES // width)))


def _split_ranges(units: list[tuple[int, int, int]], limit: int) -> list[tuple[int, int]]:
    """Group consecutive units of row numbers into ranges of at most ``limit`` rows.

    Parameters
    ----------
    units : list[tuple[int, int, int]]
        ``(start, end, rows)`` of each unit, in ascending order of row
        numbers: the row numbers from ``start`` up to ``end`` (excluded) hold
        ``rows`` merged rows.
    limit : int
        Most merged rows of one range.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` of each range, in ascending order. Neighboring
        units are grouped while their rows total at most ``limit``; a unit
        holding more than ``limit`` rows is a range of its own.
    """
    ranges: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    total = 0
    for start, end, rows in units:
        if current is not None and total + rows <= limit:
            current, total = (current[0], end), total + rows
            continue
        if current is not None:
            ranges.append(current)
        current, total = (start, end), rows
    if current is not None:
        ranges.append(current)
    return ranges


def _count_units(
    merged: pl.LazyFrame, part: pl.Expr, column: str, limit: int
) -> list[tuple[int, int, int]]:
    """Count the merged rows of one part by intervals of row numbers.

    Parameters
    ----------
    merged : pl.LazyFrame
        Merged rows of every bucket, with the row-number columns.
    part : pl.Expr
        Filter that selects the rows of the part.
    column : str
        Row-number column that orders the part.
    limit : int
        Most merged rows put back in order at a time.

    Returns
    -------
    list[tuple[int, int, int]]
        ``(start, end, rows)`` of each unit holding rows, in ascending order:
        an interval of ``_COUNT_BLOCK`` row numbers, or a single row number
        for an interval holding more than ``limit`` rows.
    """
    row = pl.col(column)
    blocks = (
        merged.filter(part)
        .group_by(row // _COUNT_BLOCK)
        .agg(pl.len())
        .sort(column)
        .collect(engine="streaming")
    )
    units: list[tuple[int, int, int]] = []
    for block, rows in blocks.iter_rows():
        start, end = block * _COUNT_BLOCK, (block + 1) * _COUNT_BLOCK
        if rows <= limit:
            units.append((start, end, rows))
            continue
        counts = (
            merged.filter(part & (row >= start) & (row < end))
            .group_by(row)
            .agg(pl.len())
            .sort(column)
            .collect(engine="streaming")
        )
        units += [(number, number + 1, n) for number, n in counts.iter_rows()]
    return units


def _restore_ranges(
    merged: pl.LazyFrame, how: HOW, left_row: str, right_row: str, limit: int
) -> list[tuple[pl.Expr, list[str]]]:
    """List the ranges of row numbers to put the merged rows back in order.

    Parameters
    ----------
    merged : pl.LazyFrame
        Merged rows of every bucket, with the row-number columns.
    how : HOW
        Join strategy of the merge.
    left_row, right_row : str
        Names of the temporary row-number columns of each side.
    limit : int
        Most merged rows of one range, from ``_restore_limit``; a single row
        number holding more rows is a range of its own.

    Returns
    -------
    list[tuple[pl.Expr, list[str]]]
        For each range, in the order of the result, the filter that selects
        its merged rows and the row-number columns to sort them by.
    """
    def ranges(part: pl.Expr, column: str, by: list[str]) -> list[tuple[pl.Expr, list[str]]]:
        row = pl.col(column)
        return [
            (part & (row >= start) & (row < end), by)
            for start, end in _split_ranges(_count_units(merged, part, column, limit), limit)
        ]

    if how == "right":
        return ranges(pl.col(right_row).is_not_null(), right_row, [right_row, left_row])
    out = ranges(pl.col(left_row).is_not_null(), left_row, [left_row, right_row])
    if how == "full":
        out += ranges(pl.col(left_row).is_null(), right_row, [right_row])
    return out


def merge_on_disk(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    columns: list[str] | None = None,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: HOW = "left",
    indicator: str | None = "_merge",
    verbose: bool = False,
    *,
    directory: str | Path | None = None,
    delete_on_exit: bool | None = None,
    buckets: int = 16,
) -> pl.LazyFrame:
    """Merge two LazyFrames like ``merge`` with little memory, on disk.

    The result is the same as ``merge`` with the same arguments: the row
    order, the coalesced keys, the column names and types, and the
    merge-origin values. It is computed with its intermediate data on disk,
    written to Parquet files, and returned as a ``pl.LazyFrame`` that reads
    them.

    Only ``pl.LazyFrame`` is accepted, and a ``pl.LazyFrame`` is always
    returned, unlike the other helpers that accept both kinds.

    With ``buckets >= 2``, both sides are split on disk into buckets by the
    hash of their keys, each bucket is merged with ``merge``, and the merged
    rows are put back in order a range of row numbers at a time. With
    ``buckets=1``, ``merge`` is written in one go with the streaming engine.

    The merge is computed completely inside this call, and the result files
    are written before it returns. A ``select`` or ``filter`` on the returned
    frame applies when the result files are read; it does not reduce the
    work of the merge.

    Parameters
    ----------
    left, right : pl.LazyFrame
        Frames to merge.
    columns : list[str] | None, default None
        Columns to take from ``right``; its key columns are always taken. If
        ``None``, every column is taken.
    on : str | list[str] | None, default None
        Key columns shared by both sides.
    left_on, right_on : str | list[str] | None, default None
        Key columns of each side, used instead of ``on``.
    how : HOW, default "left"
        Join strategy passed to polars.
    indicator : str | None, default "_merge"
        Name of the merge-origin column holding ``"both"``, ``"left_only"``,
        or ``"right_only"``. ``None`` omits the column.
    verbose : bool, default False
        Whether to print the shape change and the merge-origin counts, as
        ``merge`` does, counted from the written result.
    directory : str | Path | None, default None
        Directory to write the result in; it is created with its parents if
        missing. A relative path is taken from the current working directory
        at the call; ``~`` and environment variables are not expanded. Files
        already in it are neither read, overwritten, nor deleted: each call
        writes ``merge-<id>-00000.parquet``, ``merge-<id>-00001.parquet``,
        ... with an ``<id>`` of its own, and works in ``merge-<id>-work/``,
        which is deleted before returning. If ``None``, the same names are
        written directly in the user cache directory of the OS, the default
        directory of ``scan_parquet_relaxed``: ``%LOCALAPPDATA%`` (or
        ``~/AppData/Local``) ``/polars-helpers/cache`` on Windows,
        ``~/Library/Caches/polars-helpers`` on macOS, and
        ``$XDG_CACHE_HOME`` (or ``~/.cache``) ``/polars-helpers`` elsewhere.
        The temporary directory of the OS is never used; if the directory
        cannot be written, the error is raised and no other directory is
        tried.
    delete_on_exit : bool | None, default None
        Whether to delete the result files of this call when the process
        exits normally (``atexit``). ``None`` deletes them when
        ``directory`` is ``None`` and keeps them when it is given. Once the
        result is written, the files to delete are registered in a session
        of the process, kept as ``.polars-helpers-session-<id>.lock`` and
        ``.json`` directly in the directory and deleted with them; with a
        given ``directory``, the directories this call created are deleted
        too, the innermost first, once they are empty. The default
        directory and directories that existed before the call are kept.
        Files that cannot be deleted, for example because they are open,
        are skipped. If the process is killed, a later call of
        ``merge_on_disk`` or ``scan_parquet_relaxed`` on the same directory
        deletes the registered files and the session files. With ``False``,
        only the result files are written, and nothing deletes them.
    buckets : int, default 16
        Number of buckets to split both sides into. ``buckets=1`` is faster
        and uses less memory when ``right`` is small; keep the default when
        ``right`` is large. For example, on a 12-thread Windows machine, a
        left join of 100 million rows with 1 million rows took 3-5 s and
        1.3-1.7 GB with ``buckets=1`` against 10-13 s and 2.7-3.0 GB with
        ``buckets=16``, while a left join of 100 million rows with 100
        million String keys took 13.5 GB with ``buckets=1`` against 24 s
        and 3.1 GB with ``buckets=16``.

    Returns
    -------
    pl.LazyFrame
        Frame that reads the result files.

    Raises
    ------
    TypeError
        If ``left`` and ``right`` are not both ``pl.LazyFrame``, or if
        ``directory`` is not a ``str``, ``Path``, or ``None``.
    ValueError
        If ``buckets`` is not a positive integer, if the keys are not given
        as either ``on`` or both ``left_on`` and ``right_on``, if
        ``indicator`` names an existing column, or if ``directory`` is an
        empty string.
    NotADirectoryError
        If ``directory`` is an existing file.
    OSError
        If the directory cannot be created or written.

    Notes
    -----
    Key types that cannot be joined raise the same exception as collecting
    ``merge`` of the empty frames. Every check is done before anything is
    written. If the merge fails midway, the files and directories this call
    wrote are deleted before the exception is raised, and no session is
    created.

    With ``buckets >= 2``, the merged rows are put back in order about
    1 GiB at a time, as ``estimated_size()`` measures it; sorting them makes
    the peak about twice as large. The size of a row is measured on the
    first 10,000 merged rows, so data whose later rows are much larger, such
    as longer strings, can go beyond it. The rows that one input row gives
    are always put back in order together, so a row that matches many rows
    takes the memory of all its matches at once.

    Each call, whatever ``delete_on_exit`` is, also deletes, directly in
    the directory it uses, the files that ended processes registered there
    and left, for example because they were killed, with their session
    files. Files of living processes, unregistered files, directories, and
    other directories are left alone.
    """
    if not isinstance(left, pl.LazyFrame) or not isinstance(right, pl.LazyFrame):
        raise TypeError(
            f"left: {type(left)} and right: {type(right)} must both be polars.LazyFrame"
        )
    if isinstance(buckets, bool) or not isinstance(buckets, int) or buckets < 1:
        raise ValueError(f"buckets must be a positive integer, got {buckets!r}")

    empty = merge(
        left.clear(), right.clear(), columns, on, left_on, right_on, how, indicator
    ).collect()

    path = _disk.resolve_root(directory)
    created = _disk.make_directory(path)
    _disk.collect_stale_sessions(path)
    call_id = uuid.uuid4().hex
    work = path / f"merge-{call_id}-work"
    results: list[Path] = []

    def result_file() -> Path:
        file = path / f"merge-{call_id}-{len(results):05d}.parquet"
        results.append(file)
        return file

    try:
        left_keys, right_keys = _resolve_keys(on, left_on, right_on)
        if columns is not None:
            right = right.select(*right_keys, *(c for c in columns if c not in right_keys))
        dtypes = _common_key_dtypes(left, right, left_keys, right_keys)

        def join(left_part: pl.LazyFrame, right_part: pl.LazyFrame) -> pl.LazyFrame:
            return merge(
                _align_full_keys(left_part, left_keys, dtypes, how),
                _align_full_keys(right_part, right_keys, dtypes, how),
                left_on=left_keys,
                right_on=right_keys,
                how=how,
                indicator=indicator,
            )

        if buckets == 1:
            join(left, right).sink_parquet(result_file(), engine="streaming")
        else:
            left_columns, right_columns = left.collect_schema(), right.collect_schema()
            taken = {*left_columns, *right_columns}
            taken |= {f"{c}_right" for c in taken}
            if indicator is not None:
                taken.add(indicator)
            left_row, right_row, bucket_name = _temporary_names(
                taken, [_LEFT_ROW, _RIGHT_ROW, _BUCKET]
            )
            left_numbered = left.with_row_index(left_row)
            right_numbered = right.with_row_index(right_row)

            left_files = _partition(
                left_numbered, _bucket_expr(left_keys, dtypes, buckets, bucket_name), work / "left"
            )
            right_files = _partition(
                right_numbered, _bucket_expr(right_keys, dtypes, buckets, bucket_name), work / "right"
            )

            left_schema = left_numbered.collect_schema()
            right_schema = right_numbered.collect_schema()
            merged_dir = work / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_files: list[Path] = []
            for bucket in range(buckets):
                if bucket not in left_files and bucket not in right_files:
                    continue
                file = merged_dir / f"{bucket:05d}.parquet"
                join(
                    _scan(left_files.get(bucket, []), left_schema),
                    _scan(right_files.get(bucket, []), right_schema),
                ).sink_parquet(file, row_group_size=_ROW_GROUP_ROWS, engine="streaming")
                merged_files.append(file)

            if merged_files:
                merged = pl.scan_parquet(merged_files)
                limit = _restore_limit(merged)
                for cond, by in _restore_ranges(merged, how, left_row, right_row, limit):
                    # Sorting the collected range uses less memory than
                    # sorting in the query; the range is freed before the
                    # next one is collected.
                    part = merged.filter(cond).collect().sort(by).drop(left_row, right_row)
                    if part.height:
                        part.write_parquet(result_file())
                    del part

            shutil.rmtree(work)

        if not results:
            empty.write_parquet(result_file())

        # Registered only once every result file is written, so that a call
        # that fails, even while registering, leaves no session files.
        if delete_on_exit is None:
            delete_on_exit = directory is None
        if delete_on_exit:
            _disk.register_on_exit(path, results, created if directory is not None else ())
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        _disk.delete_paths(results, created)
        raise

    out = pl.scan_parquet(results)
    if verbose:
        _print_summary(left, out, indicator)
    return out

