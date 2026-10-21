"""Read the shape, columns, schema, and values of polars frames."""

from __future__ import annotations

import polars as pl

from .utils import natural_keys


def get_value_counts(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str | bool,
    sort: bool = True,
) -> pl.DataFrame:
    """Count occurrences of each distinct value in one column.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame containing the column to summarize.
    alias : str | bool
        Name of the column to count.
    sort : bool, default True
        Whether to sort the result by descending count.

    Returns
    -------
    pl.DataFrame
        Distinct values with their ``count``.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if isinstance(df, pl.LazyFrame):
        vc = df.select(alias).collect()[alias].value_counts()  # ty:ignore[not-subscriptable, unresolved-attribute]
    elif isinstance(df, pl.DataFrame):
        vc = df[alias].value_counts()  # ty:ignore[unresolved-attribute]
    else:
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")
    if sort:
        vc = vc.sort("count", descending=True)
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
    if isinstance(df, pl.DataFrame):
        return df.shape
    elif isinstance(df, pl.LazyFrame):
        n_rows = df.select(pl.len().alias("n_rows")).collect()["n_rows"][0]  # ty:ignore[not-subscriptable]
        n_cols = len(df.collect_schema())
        return (n_rows, n_cols)
    else:
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


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
    if isinstance(df, pl.DataFrame):
        return df.columns
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


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
        Column names mapped to their dtypes.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if args:
        df = df.select(args)
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema()
    if isinstance(df, pl.DataFrame):
        return df.schema
    raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")


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
        Names of the candidate columns with a numeric dtype.
    """
    if not args:
        args = get_columns(df)
    numeric_types = [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    mapping = get_schema(df, args)
    return [key for key, dtype in mapping.items() if dtype in numeric_types]


def get_unique_values(
    df: pl.DataFrame | pl.LazyFrame,
    alias: str,
    sort: bool = True,
) -> list[str]:
    """Return one column's distinct non-null values.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Frame containing the column.
    alias : str
        Name of the column to read.
    sort : bool, default True
        Whether to sort the values with ``natural_keys``, so that embedded
        numbers order numerically rather than lexicographically.

    Returns
    -------
    list[str]
        Distinct values of the column.

    Raises
    ------
    TypeError
        If ``df`` is neither a ``pl.DataFrame`` nor a ``pl.LazyFrame``.
    """
    if isinstance(df, pl.LazyFrame):
        uniques = df.select(pl.col(alias).drop_nulls().unique()).collect().to_series().to_list()
    elif isinstance(df, pl.DataFrame):
        uniques = df.select(pl.col(alias).drop_nulls().unique()).to_series().to_list()
    else:
        raise TypeError("df must be a polars.DataFrame or polars.LazyFrame")

    if sort:
        return sorted(uniques, key=natural_keys)
    return uniques
