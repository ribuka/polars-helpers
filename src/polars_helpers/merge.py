"""Join two polars frames with a pandas-style merge indicator."""

from __future__ import annotations

from typing import Literal

import polars as pl

from .info import get_columns, get_shape, get_value_counts

HOW = Literal["left", "right", "full", "inner"]


def _join_with_indicator(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    on: list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: HOW = "left",
    indicator: str | bool = "_merge",
) -> pl.DataFrame | pl.LazyFrame:
    """Join two polars frames and optionally add a merge-origin indicator.

    Parameters
    ----------
    left, right : pl.DataFrame | pl.LazyFrame
        Frames to join. Both must be of the same kind.
    on : list[str] | None, default None
        Shared key columns. Takes precedence over ``left_on``/``right_on``.
    left_on, right_on : str | list[str] | None, default None
        Per-side key columns, used when ``on`` is empty or ``None``.
    how : HOW, default "left"
        Join strategy passed to polars.
    indicator : str | bool, default "_merge"
        Name of the merge-origin column holding ``"both"``, ``"left_only"``,
        or ``"right_only"``. Pass ``False`` to omit the column.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Joined frame, of the same kind as the inputs.

    Raises
    ------
    ValueError
        If neither ``on`` nor both of ``left_on`` and ``right_on`` is given.
    """
    left = left.with_columns(pl.lit(True).alias("in_left"))
    right = right.with_columns(pl.lit(True).alias("in_right"))

    if on:
        out = left.join(right, on=on, how=how)  # ty:ignore[invalid-argument-type]
    elif left_on and right_on:
        out = left.join(right, left_on=left_on, right_on=right_on, how=how)  # ty:ignore[invalid-argument-type]
    else:
        raise ValueError("Invalid 'on' parameter")

    out = out.with_columns([
        pl.col("in_left").fill_null(False),
        pl.col("in_right").fill_null(False),
    ])

    if indicator is not False:
        out = out.with_columns([
            pl.when(pl.col("in_left") & pl.col("in_right")).then(pl.lit("both"))
            .when(pl.col("in_left")).then(pl.lit("left_only"))
            .otherwise(pl.lit("right_only"))
            .alias(indicator)
        ])

    out = out.drop("in_left", "in_right")

    return out


def merge(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    columns: list[str] | None = None,
    on: list[str] | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: HOW = "left",
    indicator: str | bool = "_merge",
    show_result: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """Merge two same-kind polars frames with optional column selection.

    Parameters
    ----------
    left, right : pl.DataFrame | pl.LazyFrame
        Frames to merge. Both must be of the same kind.
    columns : list[str] | None, default None
        Columns to take from ``right``. If ``None``, every column is taken.
        The list itself is not modified.
    on : list[str] | None, default None
        Shared key columns. If ``None`` or empty, ``left_on``/``right_on``
        are used instead.
    left_on, right_on : str | None, default None
        Per-side key columns, used when ``on`` is empty.
    how : HOW, default "left"
        Join strategy passed to polars.
    indicator : str | bool, default "_merge"
        Name of the merge-origin column, or ``False`` to omit it.
    show_result : bool, default True
        Whether to print the shape change and the merge-origin counts, which
        is intended for interactive notebook use.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Merged frame, of the same kind as the inputs.

    Raises
    ------
    TypeError
        If ``left`` and ``right`` are not of the same kind.
    ValueError
        If ``on`` is empty and ``left_on``/``right_on`` are not both given.
    """
    if type(left) is not type(right):
        raise TypeError(f"left: {type(left)} and right: {type(right)} must be the same type")

    if (
        isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame)
    ) or (
        isinstance(left, pl.LazyFrame) and isinstance(right, pl.LazyFrame)
    ):
        if show_result:
            _shape = get_shape(left)

        if columns is None:
            columns = get_columns(right)

        if on:
            left = _join_with_indicator(
                left, right.select(columns), on=on, how=how, indicator=indicator,
            )

        else:
            if not left_on or not right_on:
                raise ValueError("`left_on` and `right_on` must be specified if `on` is None")

            if right_on not in columns:
                columns = [*columns, right_on]

            left = _join_with_indicator(
                left, right.select(columns), on=[], left_on=left_on, right_on=right_on, how=how, indicator=indicator,
            )

        if show_result:
            print(f"{_shape} -> {get_shape(left)}")
            if indicator is not False:
                print(get_value_counts(
                    left, alias=indicator,
                ))

    return left
