"""Drop columns that are mostly missing or that writers cannot serialize."""

from __future__ import annotations

import polars as pl
import polars.selectors as cs

from .info import get_schema


def drop_list_type_columns(
    df: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame | pl.LazyFrame:
    """Drop every List-typed column, which most writers cannot serialize.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame to inspect.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        ``df`` without its List-typed columns. The dropped names are printed
        so notebook callers can see what was removed.
    """
    drop_list = [
        column
        for column, dtype in get_schema(df).items()
        if dtype == pl.datatypes.List
    ]
    print(f"{drop_list = }")

    return df.drop(drop_list)


def validate_missing_ratio_threshold(threshold: float) -> None:
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


def _count_missing_values(
    frame: pl.DataFrame | pl.LazyFrame,
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
    frame : pl.DataFrame | pl.LazyFrame
        Frame whose columns are evaluated. A ``pl.LazyFrame`` collects only
        the statistics queries.
    include_nan_missing : bool
        Whether NaN values count as missing for floating-point columns.

    Returns
    -------
    tuple[int, dict[str, int]]
        Row count and missing-value count per column, in column order.
    """
    nan_count_expr = cs.float().is_nan().sum()
    if isinstance(frame, pl.LazyFrame):
        queries = [frame.select(pl.len()), frame.null_count()]
        if include_nan_missing:
            queries.append(frame.select(nan_count_expr))
        stats = pl.collect_all(queries)
    else:
        stats = [frame.select(pl.len()), frame.null_count()]
        if include_nan_missing:
            stats.append(frame.select(nan_count_expr))
    row_count = stats[0].item()
    missing_counts, *nan_counts = [
        stat.row(0, named=True) if stat.width else {} for stat in stats[1:]
    ]
    for col_name, nan_count in (nan_counts[0] if nan_counts else {}).items():
        missing_counts[col_name] += nan_count
    return row_count, missing_counts


def drop_all_null_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    include_nan_missing: bool = True,
    threshold: float = 0.99,
) -> pl.DataFrame | pl.LazyFrame:
    """Drop columns whose missing-value ratio exceeds a threshold.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Frame whose columns are evaluated for missing values.
    include_nan_missing : bool, default True
        Whether NaN values count as missing for floating-point columns.
    threshold : float, default 0.99
        Inclusive maximum missing-value ratio for retained columns.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        ``frame`` restricted to columns at or below ``threshold``.

    Raises
    ------
    ValueError
        If ``threshold`` is not between 0.0 and 1.0.
    """
    validate_missing_ratio_threshold(threshold)

    row_count, missing_counts = _count_missing_values(frame, include_nan_missing)
    if row_count == 0:
        return frame

    keep_cols = [
        col_name
        for col_name, missing_count in missing_counts.items()
        if missing_count / row_count <= threshold
    ]

    return frame.select(keep_cols)
