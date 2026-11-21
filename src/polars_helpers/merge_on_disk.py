"""Merge two large polars LazyFrames with little memory by working on disk."""

from __future__ import annotations

import atexit
import shutil
import tempfile
import uuid
from pathlib import Path

import polars as pl

from .merge import HOW, _print_summary, _resolve_keys, merge

# Temporary row numbers of each side; named so they do not clash with the
# columns of the frames being merged.
_LEFT_ROW = "__polars_helpers_left_row"
_RIGHT_ROW = "__polars_helpers_right_row"

# Name of the partition key that holds the bucket of each row.
_BUCKET = "__polars_helpers_bucket"

# Fixed seed of the key hash, so both sides put equal keys in one bucket.
_HASH_SEED = 0

# Number of row numbers whose rows are put back in order at a time.
_RESTORE_ROWS = 10_000_000

# Rows per row group of the merged buckets, so that putting one range of
# row numbers back in order reads only the row groups of that range.
_ROW_GROUP_ROWS = 1_000_000


def _bucket_expr(keys: list[str], dtypes: list[pl.DataType], buckets: int) -> pl.Expr:
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
    return (pl.struct(fields).hash(seed=_HASH_SEED) % buckets).alias(_BUCKET)


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


def _row_count(files: list[Path]) -> int:
    """Count the rows of Parquet files from their metadata.

    Parameters
    ----------
    files : list[Path]
        Files to count.

    Returns
    -------
    int
        Total number of rows; ``0`` if there are no files.
    """
    if not files:
        return 0
    return pl.scan_parquet(files).select(pl.len()).collect().item()


def _restore_ranges(
    how: HOW, left_rows: int, right_rows: int
) -> list[tuple[pl.Expr, list[str]]]:
    """List the ranges of row numbers to put the merged rows back in order.

    Parameters
    ----------
    how : HOW
        Join strategy of the merge.
    left_rows, right_rows : int
        Row counts of each side.

    Returns
    -------
    list[tuple[pl.Expr, list[str]]]
        For each range, in the order of the result, the filter that selects
        its merged rows and the row-number columns to sort them by.
    """
    left_row, right_row = pl.col(_LEFT_ROW), pl.col(_RIGHT_ROW)

    def ranges(rows: int, column: pl.Expr) -> list[pl.Expr]:
        return [
            (column >= start) & (column < start + _RESTORE_ROWS)
            for start in range(0, rows, _RESTORE_ROWS)
        ]

    if how == "right":
        return [(cond, [_RIGHT_ROW, _LEFT_ROW]) for cond in ranges(right_rows, right_row)]
    out = [(cond, [_LEFT_ROW, _RIGHT_ROW]) for cond in ranges(left_rows, left_row)]
    if how == "full":
        out += [
            (left_row.is_null() & cond, [_RIGHT_ROW])
            for cond in ranges(right_rows, right_row)
        ]
    return out


def _delete(files: list[Path], directories: list[Path]) -> None:
    """Delete files, then the directories that are left empty.

    Files and directories that cannot be deleted, for example because they
    are open, are skipped without raising.

    Parameters
    ----------
    files : list[Path]
        Files to delete.
    directories : list[Path]
        Directories to delete if empty, the innermost first.
    """
    for file in files:
        try:
            file.unlink(missing_ok=True)
        except OSError:
            pass
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def _make_directory(directory: str | Path | None) -> tuple[Path, list[Path]]:
    """Create the directory to write the result in.

    Parameters
    ----------
    directory : str | Path | None
        Directory given by the caller, or ``None`` for a new temporary one.

    Returns
    -------
    tuple[Path, list[Path]]
        The directory, and the directories this call created, the innermost
        first.
    """
    if directory is None:
        path = Path(tempfile.mkdtemp(prefix="polars_helpers_merge_"))
        return path, [path]
    path = Path(directory)
    created: list[Path] = []
    ancestor = path
    while not ancestor.exists():
        created.append(ancestor)
        ancestor = ancestor.parent
    path.mkdir(parents=True, exist_ok=True)
    return path, created


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
        missing. Files already in it are neither read, overwritten, nor
        deleted: each call writes ``merge-<id>-00000.parquet``,
        ``merge-<id>-00001.parquet``, ... with an ``<id>`` of its own, and
        works in ``merge-<id>-work/``, which is deleted before returning.
        If ``None``, a new directory is created under the temporary
        directory of the OS with ``tempfile.mkdtemp``.
    delete_on_exit : bool | None, default None
        Whether to delete the result files of this call when the process
        exits normally (``atexit``), together with the directories this call
        created once they are empty. ``None`` deletes them when
        ``directory`` is ``None`` and keeps them when it is given. Files
        that cannot be deleted, for example because they are open, are
        skipped. Nothing is deleted when the process is killed.
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
        If ``left`` and ``right`` are not both ``pl.LazyFrame``.
    ValueError
        If ``buckets`` is not a positive integer, if the keys are not given
        as either ``on`` or both ``left_on`` and ``right_on``, or if
        ``indicator`` names an existing column.
    NotADirectoryError
        If ``directory`` is an existing file.

    Notes
    -----
    Key types that cannot be joined raise the same exception as collecting
    ``merge`` of the empty frames. Every check is done before anything is
    written. If the merge fails midway, the files and directories this call
    wrote are deleted before the exception is raised.
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

    if directory is not None and Path(directory).exists() and not Path(directory).is_dir():
        raise NotADirectoryError(f"directory is not a directory: {directory}")

    path, created = _make_directory(directory)
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
            left_numbered = left.with_row_index(_LEFT_ROW)
            right_numbered = right.with_row_index(_RIGHT_ROW)

            left_files = _partition(
                left_numbered, _bucket_expr(left_keys, dtypes, buckets), work / "left"
            )
            right_files = _partition(
                right_numbered, _bucket_expr(right_keys, dtypes, buckets), work / "right"
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
                left_rows = _row_count([f for fs in left_files.values() for f in fs])
                right_rows = _row_count([f for fs in right_files.values() for f in fs])
                for cond, by in _restore_ranges(how, left_rows, right_rows):
                    # Sorting the collected range uses less memory than
                    # sorting in the query; the range is freed before the
                    # next one is collected.
                    part = merged.filter(cond).collect().sort(by).drop(_LEFT_ROW, _RIGHT_ROW)
                    if part.height:
                        part.write_parquet(result_file())
                    del part

            shutil.rmtree(work)

        if not results:
            empty.write_parquet(result_file())
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        _delete(results, created)
        raise

    if delete_on_exit is None:
        delete_on_exit = directory is None
    if delete_on_exit:
        atexit.register(_delete, list(results), created)

    out = pl.scan_parquet(results)
    if verbose:
        _print_summary(left, out, indicator)
    return out

