"""Read the shape, columns, schema, and values of polars frames."""

from __future__ import annotations

from typing import Any

import polars as pl
import polars.selectors as cs

from .utils import natural_keys


def _require_frame(df: object) -> None:
    """Reject inputs that are not polars frames.

    Parameters
    ----------
    df : object
        Value expected to be a ``pl.DataFrame`` or ``pl.LazyFrame``.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


def _collect(
    df: pl.DataFrame | pl.LazyFrame,
    *exprs: str | pl.Expr,
) -> pl.DataFrame:
    """Select expressions from a frame and return them as a DataFrame.

    Only the selected expressions are evaluated, so a ``LazyFrame`` is never
    materialized in full.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame to select from.
    *exprs : str | pl.Expr
        Column names or expressions to select.

    Returns
    -------
    pl.DataFrame
        The selected columns, collected if ``df`` is a ``LazyFrame``.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    _require_frame(df)
    selected = df.select(*exprs)
    if isinstance(selected, pl.LazyFrame):
        return selected.collect()
    return selected


def _value_counts_query(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str,
    sort: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """Build the per-value counts of one column without collecting it.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame containing the column to summarize.
    alias : str
        Name of the column to count.
    sort : bool, default True
        Whether to sort by descending count, breaking ties by value.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Distinct values with their ``count``, of the same kind as ``df``; a
        ``LazyFrame`` stays an unevaluated query.
    """
    vc = df.group_by(cs.by_name(alias)).len(name="count")
    if sort:
        vc = vc.sort([pl.col("count"), cs.by_name(alias)], descending=[True, False])
    return vc


def get_value_counts(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str,
    sort: bool = True,
) -> pl.DataFrame:
    """Count occurrences of each distinct value in one column.

    A ``LazyFrame`` is aggregated before it is collected, so only one row per
    distinct value is materialized.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame containing the column to summarize.
    alias : str
        Name of the column to count.
    sort : bool, default True
        Whether to sort the result by descending count, breaking ties by
        value so that the order is stable.

    Returns
    -------
    pl.DataFrame
        Distinct values with their ``count``.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    _require_frame(df)
    vc = _value_counts_query(df, alias, sort)
    if isinstance(vc, pl.LazyFrame):
        return vc.collect()
    return vc


def get_shape(
    df: pl.DataFrame | pl.LazyFrame
) -> tuple[int, int]:
    """Return a frame's row and column counts.

    For a ``LazyFrame`` the row count is obtained from a ``pl.len()``
    aggregation and the column count from the resolved schema, so the frame
    itself is never materialized.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame to measure.

    Returns
    -------
    tuple[int, int]
        Number of rows and number of columns.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    return (_collect(df, pl.len()).item(), len(get_schema(df)))


def get_columns(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Return column names without materializing a LazyFrame.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame whose schema supplies the column names.

    Returns
    -------
    list[str]
        Frame column names in their existing order.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    return get_schema(df).names()


def get_schema(
    df: pl.DataFrame | pl.LazyFrame,
    args: list[str] | None = None,
) -> dict[str, pl.DataType]:
    """Return the schema of a polars frame, optionally limited to columns.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame whose schema is resolved. A ``LazyFrame`` is not materialized.
    args : list[str] | None, default None
        Columns to restrict the schema to. If ``None`` or empty, the full
        schema is returned.

    Returns
    -------
    dict[str, pl.DataType]
        Column names mapped to their dtypes, in the order of ``args`` when
        given.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    polars.exceptions.ColumnNotFoundError
        If a name in ``args`` is not a column of ``df``.
    """
    _require_frame(df)
    schema = df.collect_schema()
    if not args:
        return schema
    missing = [name for name in args if name not in schema]
    if missing:
        raise pl.exceptions.ColumnNotFoundError(f"columns not found: {missing}")
    return pl.Schema({name: schema[name] for name in args})


def get_numeric_args(
    df: pl.DataFrame,
    args: list[str] | None = None,
) -> list[str]:
    """Return numeric column names from a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Frame whose schema is inspected.
    args : list[str] | None, default None
        Candidate column names. If ``None`` or empty, every column is
        considered.

    Returns
    -------
    list[str]
        Names of the candidate columns with a numeric dtype (integer,
        unsigned integer, floating point, or decimal).
    """
    return [key for key, dtype in get_schema(df, args).items() if dtype.is_numeric()]


def get_unique_values(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str,
    sort: bool = True,
) -> list[Any]:
    """Return one column's distinct non-null values.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame containing the column.
    alias : str
        Name of the column to read.
    sort : bool, default True
        Whether to sort the values. String columns are sorted with
        ``natural_keys``, so that embedded numbers order numerically rather
        than lexicographically; other columns are sorted by polars.

    Returns
    -------
    list[Any]
        Distinct values of the column, as Python objects.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    is_string = get_schema(df, [alias])[alias] == pl.String
    expr = cs.by_name(alias).drop_nulls().unique()
    if sort and not is_string:
        expr = expr.sort()
    values = _collect(df, expr).to_series().to_list()

    if sort and is_string:
        return sorted(values, key=natural_keys)
    return values
