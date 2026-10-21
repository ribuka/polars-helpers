"""Drop columns that are mostly missing or that writers cannot serialize."""

from __future__ import annotations

import polars as pl
import polars.selectors as cs

from .info import get_schema


def drop_nested_type_columns[FrameT: (pl.DataFrame, pl.LazyFrame)](
    df: FrameT, verbose: bool = False
) -> FrameT:
    """Drop every nested-typed column, which flat writers cannot serialize.

    Nested dtypes are ``List``, ``Array``, and ``Struct``.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame to inspect. A ``LazyFrame`` is not materialized.
    verbose : bool, default False
        Whether to print the names of the dropped columns, which is intended
        for interactive notebook use.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        ``df`` without its nested-typed columns, of the same kind as ``df``.
    """
    drop_cols = [
        column for column, dtype in get_schema(df).items() if dtype.is_nested()
    ]
    if verbose:
        print(f"{drop_cols = }")

    return df.drop(drop_cols)


def _validate_missing_ratio_threshold(threshold: float) -> None:
    """Validate a missing-value ratio threshold.

    Parameters
    ----------
    threshold : float
        Inclusive missing-value ratio threshold to validate.

    Raises
    ------
    ValueError
        If ``threshold`` is not between 0.0 and 1.0.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")


def _missing_value_queries[FrameT: (pl.DataFrame, pl.LazyFrame)](
    df: FrameT, include_nan_missing: bool
) -> list[FrameT]:
    """Build the row-count and missing-value queries of ``df``.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame whose columns are evaluated.
    include_nan_missing : bool
        Whether to add a query counting NaNs of the floating-point columns.

    Returns
    -------
    list[pl.DataFrame] | list[pl.LazyFrame]
        Row count, null counts, and optionally NaN counts, each a one-row
        frame of the same kind as ``df``.
    """
    queries = [df.select(pl.len()), df.null_count()]
    if include_nan_missing:
        queries.append(df.select(cs.float().is_nan().sum()))
    return queries


def _count_missing_values[FrameT: (pl.DataFrame, pl.LazyFrame)](
    df: FrameT,
    include_nan_missing: bool,
) -> tuple[int, dict[str, int]]:
    """Count the rows and each column's missing values in bulk.

    Nulls are counted by ``null_count`` and, when requested, NaNs by one
    selector over the floating-point columns, instead of one aggregation
    expression per column, which dominates the cost at six-figure widths.
    ``is_nan`` yields null for null entries and ``sum`` skips them, so no
    entry is counted twice. A ``pl.DataFrame`` is aggregated eagerly,
    because routing it through a lazy query costs far more than the
    aggregation itself on wide frames.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame whose columns are evaluated. A ``pl.LazyFrame`` collects only
        the statistics queries.
    include_nan_missing : bool
        Whether NaN values count as missing for floating-point columns.

    Returns
    -------
    tuple[int, dict[str, int]]
        Row count and missing-value count per column, in column order.
    """
    if isinstance(df, pl.LazyFrame):
        stats = pl.collect_all(_missing_value_queries(df, include_nan_missing))
    else:
        stats = _missing_value_queries(df, include_nan_missing)

    row_count = stats[0].item()
    missing_counts: dict[str, int] = {}
    for stat in stats[1:]:
        for col_name, count in (stat.row(0, named=True) if stat.width else {}).items():
            missing_counts[col_name] = missing_counts.get(col_name, 0) + count
    return row_count, missing_counts


def drop_all_null_columns[FrameT: (pl.DataFrame, pl.LazyFrame)](
    df: FrameT,
    include_nan_missing: bool = True,
    threshold: float = 0.99,
) -> FrameT:
    """Drop columns whose missing-value ratio exceeds a threshold.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame whose columns are evaluated for missing values.
    include_nan_missing : bool, default True
        Whether NaN values count as missing for floating-point columns.
    threshold : float, default 0.99
        Inclusive maximum missing-value ratio for retained columns.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        ``df`` restricted to columns at or below ``threshold``, of the same
        kind as ``df``.

    Raises
    ------
    ValueError
        If ``threshold`` is not between 0.0 and 1.0.
    """
    _validate_missing_ratio_threshold(threshold)

    row_count, missing_counts = _count_missing_values(df, include_nan_missing)
    if row_count == 0:
        return df

    keep_cols = [
        col_name
        for col_name, missing_count in missing_counts.items()
        if missing_count / row_count <= threshold
    ]
    if len(keep_cols) == len(missing_counts):
        return df

    return df.select(keep_cols)
