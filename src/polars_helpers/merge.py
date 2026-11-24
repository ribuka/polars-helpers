"""Join two polars frames with a pandas-style merge indicator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import polars as pl

from .info import _value_counts_query, get_columns

if TYPE_CHECKING:
    from polars._typing import EngineType

HOW = Literal["left", "right", "full", "inner"]

# Key relations ``merge`` can check, as with ``polars.DataFrame.join``: the
# side written ``1`` must have unique keys, and ``"m:m"`` checks nothing.
VALIDATE = Literal["m:m", "1:m", "m:1", "1:1"]

# Most duplicated key values shown in the error of a failed ``validate``.
_EXAMPLES = 5

# Temporary flags marking the rows of each side; named so they do not clash
# with the columns of the frames being merged.
_IN_LEFT = "__polars_helpers_in_left"
_IN_RIGHT = "__polars_helpers_in_right"

# Row order of each join strategy, as with pandas.merge: the rows follow the
# left frame (the right frame for how="right"), and the rows sharing one row
# of that frame follow the other frame. pandas sorts the keys of an outer
# join; a full join here keeps the left order and appends the right-only rows
# in the right order instead.
_MAINTAIN_ORDER: dict[HOW, Literal["left_right", "right_left"]] = {
    "left": "left_right",
    "right": "right_left",
    "full": "left_right",
    "inner": "left_right",
}


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


def _check_validate(validate: object) -> None:
    """Reject a ``validate`` value that is not one of ``VALIDATE``.

    Parameters
    ----------
    validate : object
        Value given as ``validate``.

    Raises
    ------
    ValueError
        If ``validate`` is not one of ``"m:m"``, ``"1:m"``, ``"m:1"``, and
        ``"1:1"``.
    """
    values = get_args(VALIDATE)
    if not isinstance(validate, str) or validate not in values:
        raise ValueError(
            f"validate must be one of {', '.join(map(repr, values))}, got {validate!r}"
        )


def _unique_sides(validate: VALIDATE) -> tuple[bool, bool]:
    """Tell which sides must have unique keys.

    Parameters
    ----------
    validate : VALIDATE
        Key relation to check.

    Returns
    -------
    tuple[bool, bool]
        Whether the left keys and whether the right keys must be unique.
    """
    return validate in ("1:m", "1:1"), validate in ("m:1", "1:1")


def _count_name(keys: list[str]) -> str:
    """Choose the name of the count column of the duplicated keys.

    Parameters
    ----------
    keys : list[str]
        Key columns of one side.

    Returns
    -------
    str
        ``"count"``, or the first of ``"count_0"``, ``"count_1"``, ... that
        is not a key column.
    """
    name, n = "count", 0
    while name in keys:
        name, n = f"count_{n}", n + 1
    return name


def _duplicate_key_queries(
    frame: pl.LazyFrame, keys: list[str]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Build the number and the first examples of the duplicated keys.

    Rows with a null in any key column are left out: they match no row, so
    they never multiply rows.

    Parameters
    ----------
    frame : pl.LazyFrame
        One side of the merge.
    keys : list[str]
        Key columns of that side.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame]
        Unevaluated queries of a one-value frame with the number of key
        values that appear more than once, and of the first ``_EXAMPLES`` of
        them by key with their number of rows, in the column ``_count_name``
        names.
    """
    count = _count_name(keys)
    duplicated = (
        frame.select(keys)
        .drop_nulls()
        .group_by(keys)
        .len(name=count)
        .filter(pl.col(count) > 1)
    )
    return duplicated.select(pl.len()), duplicated.sort(keys).head(_EXAMPLES)


def _first_examples(examples: list[pl.DataFrame], keys: list[str]) -> pl.DataFrame:
    """Merge the examples of duplicated keys found in parts of one side.

    Parameters
    ----------
    examples : list[pl.DataFrame]
        Examples from ``_duplicate_key_queries`` of each part, at least one;
        each key value is in one part only.
    keys : list[str]
        Key columns of that side.

    Returns
    -------
    pl.DataFrame
        The first ``_EXAMPLES`` examples of all the parts by key.
    """
    return pl.concat(examples).sort(keys).head(_EXAMPLES)


def _raise_if_not_unique(
    validate: VALIDATE, checks: list[tuple[str, list[str], int, pl.DataFrame]]
) -> None:
    """Raise if a side that must have unique keys has duplicated keys.

    Parameters
    ----------
    validate : VALIDATE
        Key relation checked.
    checks : list[tuple[str, list[str], int, pl.DataFrame]]
        For each side checked, left first: its name (``"left"`` or
        ``"right"``), its key columns, the number of key values that appear
        more than once, and the first examples of them by key.

    Raises
    ------
    ValueError
        If any side checked has duplicated keys; the message names every
        such side with its count and examples.
    """
    failed = [check for check in checks if check[2]]
    if not failed:
        return
    sides = " and ".join(side for side, *_ in failed)
    lines = [f"validate={validate!r}: the keys of {sides} are not unique"]
    for side, keys, n, examples in failed:
        lines.append(
            f"{side} keys {keys}: {n} key values appear more than once;"
            f" the first {examples.height} by key:"
        )
        lines.append(str(examples))
    raise ValueError("\n".join(lines))


def _check_unique_keys[FrameT: (pl.DataFrame, pl.LazyFrame)](
    left: FrameT,
    right: FrameT,
    left_keys: list[str],
    right_keys: list[str],
    validate: VALIDATE,
    engine: EngineType = "auto",
) -> None:
    """Check that the keys of the sides ``validate`` names are unique.

    Only the counts and the examples of the duplicated keys are collected,
    all in one ``pl.collect_all``; ``"m:m"`` collects nothing.

    Parameters
    ----------
    left, right : pl.DataFrame | pl.LazyFrame
        Frames to merge, of the same kind.
    left_keys, right_keys : list[str]
        Key columns of each side.
    validate : VALIDATE
        Key relation to check.
    engine : EngineType, default "auto"
        Engine that collects the queries.

    Raises
    ------
    ValueError
        If a side that must have unique keys has duplicated keys.
    """
    unique_left, unique_right = _unique_sides(validate)
    sides: list[tuple[str, FrameT, list[str]]] = []
    if unique_left:
        sides.append(("left", left, left_keys))
    if unique_right:
        sides.append(("right", right, right_keys))
    if not sides:
        return
    queries = [
        q for _, frame, keys in sides for q in _duplicate_key_queries(frame.lazy(), keys)
    ]
    results = pl.collect_all(queries, engine=engine)
    _raise_if_not_unique(
        validate,
        [
            (name, keys, results[2 * i].item(), results[2 * i + 1])
            for i, (name, _, keys) in enumerate(sides)
        ],
    )


def _with_flag[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT, name: str
) -> FrameT:
    """Add a column of ``True`` that marks the rows of one join side.

    Parameters
    ----------
    frame : pl.DataFrame | pl.LazyFrame
        Frame to mark.
    name : str
        Name of the flag column.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        ``frame`` with the flag column, of the same kind as ``frame``.
    """
    return frame.with_columns(pl.lit(True).alias(name))


def _join_with_indicator[FrameT: (pl.DataFrame, pl.LazyFrame)](
    left: FrameT,
    right: FrameT,
    left_on: list[str],
    right_on: list[str],
    how: HOW,
    indicator: str | None,
) -> FrameT:
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
        sides coalesced into the left key columns, in the row order of
        ``_MAINTAIN_ORDER``.
    """
    if indicator is None:
        return left.join(
            right,
            left_on=left_on,
            right_on=right_on,
            how=how,
            coalesce=True,
            maintain_order=_MAINTAIN_ORDER[how],
        )

    out = _with_flag(left, _IN_LEFT).join(
        _with_flag(right, _IN_RIGHT),
        left_on=left_on,
        right_on=right_on,
        how=how,
        coalesce=True,
        maintain_order=_MAINTAIN_ORDER[how],
    )

    in_left = pl.col(_IN_LEFT).fill_null(False)
    in_right = pl.col(_IN_RIGHT).fill_null(False)
    out = out.with_columns(
        pl.when(in_left & in_right).then(pl.lit("both"))
        .when(in_left).then(pl.lit("left_only"))
        .otherwise(pl.lit("right_only"))
        .alias(indicator)
    )

    return out.drop(_IN_LEFT, _IN_RIGHT)


def _print_summary[FrameT: (pl.DataFrame, pl.LazyFrame)](
    left: FrameT,
    out: FrameT,
    indicator: str | None,
) -> None:
    """Print the shape change of a merge and its merge-origin counts.

    The merged ``LazyFrame`` is evaluated once: its row count is the sum of
    the merge-origin counts, or a ``pl.len()`` aggregation when there is no
    indicator, and both queries are collected together with the left row
    count.

    Parameters
    ----------
    left : pl.DataFrame | pl.LazyFrame
        Left input of the merge.
    out : pl.DataFrame | pl.LazyFrame
        Merged frame, of the same kind as ``left``.
    indicator : str | None
        Name of the merge-origin column of ``out``, or ``None`` if absent.
    """
    out_query = (
        out.select(pl.len()) if indicator is None
        else _value_counts_query(out, indicator)
    )
    queries = [left.select(pl.len()), out_query]
    left_len, out_stat = (
        pl.collect_all(queries) if isinstance(out, pl.LazyFrame) else queries
    )

    out_rows = out_stat.item() if indicator is None else out_stat["count"].sum()
    print(f"{(left_len.item(), len(get_columns(left)))} -> {(out_rows, len(get_columns(out)))}")
    if indicator is not None:
        print(out_stat)


def merge[FrameT: (pl.DataFrame, pl.LazyFrame)](
    left: FrameT,
    right: FrameT,
    columns: list[str] | None = None,
    on: str | list[str] | None = None,
    left_on: str | list[str] | None = None,
    right_on: str | list[str] | None = None,
    how: HOW = "left",
    indicator: str | None = "_merge",
    validate: VALIDATE = "m:m",
    verbose: bool = False,
) -> FrameT:
    """Merge two same-kind polars frames like ``pandas.merge``.

    The keys of both sides are coalesced into one set of key columns, named
    after the left keys, and a merge-origin column is added as with
    ``pandas.merge(..., indicator=True)``.

    The row order is kept as with ``pandas.merge``, for ``pl.DataFrame`` and
    ``pl.LazyFrame`` alike: ``"left"`` and ``"inner"`` follow the left rows,
    ``"right"`` follows the right rows, and the rows sharing one row of that
    side follow the other side. ``"full"`` follows the left rows and then
    the right-only rows in the right order, instead of sorting the keys as
    pandas does.

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
    validate : VALIDATE, default "m:m"
        Key relation to check before merging, whatever ``how`` is: ``"1:m"``
        requires unique left keys, ``"m:1"`` unique right keys, ``"1:1"``
        both, and ``"m:m"`` checks nothing. Rows with a null in any key
        column are left out, as they match no row. The check runs in this
        call, for ``pl.LazyFrame`` too: only the counts and the first five
        duplicated keys of each side checked are collected, which reads the
        inputs once more, and the returned ``pl.LazyFrame`` stays
        unevaluated.
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
        ``right_on``, if ``indicator`` names an existing column, if
        ``validate`` is not one of ``VALIDATE``, or if the keys of a side
        that ``validate`` requires to be unique are not; the message names
        each such side, its number of duplicated key values, and the first
        five of them by key with their number of rows.
    """
    if not isinstance(left, pl.DataFrame | pl.LazyFrame) or type(left) is not type(right):
        raise TypeError(
            f"left: {type(left)} and right: {type(right)} must be the same type,"
            " either polars.DataFrame or polars.LazyFrame"
        )
    _check_validate(validate)

    left_keys, right_keys = _resolve_keys(on, left_on, right_on)

    if columns is not None:
        right = right.select(*right_keys, *(c for c in columns if c not in right_keys))

    if indicator is not None and indicator in {*get_columns(left), *get_columns(right)}:
        raise ValueError(f"indicator column {indicator!r} already exists")

    _check_unique_keys(left, right, left_keys, right_keys, validate)

    out = _join_with_indicator(left, right, left_keys, right_keys, how, indicator)

    if verbose:
        _print_summary(left, out, indicator)

    return out
