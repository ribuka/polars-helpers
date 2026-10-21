"""Join two polars frames with a pandas-style merge indicator."""

from __future__ import annotations

from typing import Literal

import polars as pl

from .info import get_columns, get_shape, get_value_counts

HOW = Literal["left", "right", "full", "inner"]

# Temporary flags marking the rows of each side; named so they do not clash
# with the columns of the frames being merged.
_IN_LEFT = "__polars_helpers_in_left"
_IN_RIGHT = "__polars_helpers_in_right"


def _as_list(keys: str | list[str] | None) -> list[str]:
    """Normalize a key specification to a list of column names.

    Parameters
    ----------
    keys : str | list[str] | None
        One column name, several column names, or ``None``.

    Returns
    -------
    list[str]
        The column names; empty for ``None``.
    """
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    return list(keys)


def _resolve_keys(
    on: str | list[str] | None,
    left_on: str | list[str] | None,
    right_on: str | list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve the key columns of each side.

    Parameters
    ----------
    on : str | list[str] | None
        Key columns shared by both sides.
    left_on, right_on : str | list[str] | None
        Key columns of each side, used when ``on`` is not given.

    Returns
    -------
    tuple[list[str], list[str]]
        Key columns of the left side and of the right side.

    Raises
    ------
    ValueError
        If ``on`` is given together with ``left_on``/``right_on``, or if
        ``on`` is not given and ``left_on``/``right_on`` are not both given.
    """
    if on:
        if left_on or right_on:
            raise ValueError("pass either `on` or `left_on` and `right_on`, not both")
        keys = _as_list(on)
        return keys, keys
    if not left_on or not right_on:
        raise ValueError("`left_on` and `right_on` must be specified if `on` is None")
    return _as_list(left_on), _as_list(right_on)


def _join_with_indicator(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    left_on: list[str],
    right_on: list[str],
    how: HOW,
    indicator: str | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Join two polars frames and optionally add a merge-origin indicator.

    Parameters
    ----------
    left, right : pl.DataFrame | pl.LazyFrame
        Frames to join. Both must be of the same kind.
    left_on, right_on : list[str]
        Key columns of each side.
    how : HOW
        Join strategy passed to polars.
    indicator : str | None
        Name of the merge-origin column holding ``"both"``, ``"left_only"``,
        or ``"right_only"``. ``None`` omits the column.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Joined frame, of the same kind as the inputs, with the keys of both
        sides coalesced into the left key columns.
    """
    out = left.with_columns(pl.lit(True).alias(_IN_LEFT)).join(
        right.with_columns(pl.lit(True).alias(_IN_RIGHT)),
        left_on=left_on,
        right_on=right_on,
        how=how,
        coalesce=True,
    )

    if indicator is not None:
        in_left = pl.col(_IN_LEFT).fill_null(False)
        in_right = pl.col(_IN_RIGHT).fill_null(False)
        out = out.with_columns(
            pl.when(in_left & in_right).then(pl.lit("both"))
            .when(in_left).then(pl.lit("left_only"))
            .otherwise(pl.lit("right_only"))
            .alias(indicator)
        )

    return out.drop(_IN_LEFT, _IN_RIGHT)


def merge(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    columns: list[str] | None = None,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: HOW = "left",
    indicator: str | None = "_merge",
    verbose: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Merge two same-kind polars frames like ``pandas.merge``.

    The keys of both sides are coalesced into one set of key columns, named
    after the left keys, and a merge-origin column is added as with
    ``pandas.merge(..., indicator=True)``.

    Parameters
    ----------
    left, right : pl.DataFrame | pl.LazyFrame
        Frames to merge. Both must be of the same kind.
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
        Whether to print the shape change and the merge-origin counts, which
        is intended for interactive notebook use.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Merged frame, of the same kind as the inputs.

    Raises
    ------
    TypeError
        If ``left`` and ``right`` are not both ``pl.DataFrame`` or both
        ``pl.LazyFrame``.
    ValueError
        If the keys are not given as either ``on`` or both ``left_on`` and
        ``right_on``, or if ``indicator`` names an existing column.
    """
    if not isinstance(left, pl.DataFrame | pl.LazyFrame) or type(left) is not type(right):
        raise TypeError(
            f"left: {type(left)} and right: {type(right)} must be the same type,"
            " either polars.DataFrame or polars.LazyFrame"
        )

    left_keys, right_keys = _resolve_keys(on, left_on, right_on)

    if columns is None:
        columns = get_columns(right)
    right = right.select(*right_keys, *(c for c in columns if c not in right_keys))

    if indicator is not None and indicator in {*get_columns(left), *get_columns(right)}:
        raise ValueError(f"indicator column {indicator!r} already exists")

    out = _join_with_indicator(left, right, left_keys, right_keys, how, indicator)

    if verbose:
        print(f"{get_shape(left)} -> {get_shape(out)}")
        if indicator is not None:
            print(get_value_counts(out, alias=indicator))

    return out
