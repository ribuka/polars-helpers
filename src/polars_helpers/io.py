"""Read and write CSV files with polars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .cleaning import drop_nested_type_columns
from .utils.csv_header import detect_csv_skip_lines


def _with_skip_lines(source: str | Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add the detected ``skip_lines`` to the keyword arguments of a CSV reader.

    Parameters
    ----------
    source : str | Path
        Path of the CSV file.
    kwargs : dict[str, Any]
        Keyword arguments for ``pl.read_csv`` or ``pl.scan_csv``. Their
        ``separator`` and ``quote_char`` are used for the detection.

    Returns
    -------
    dict[str, Any]
        ``kwargs`` unchanged when ``skip_lines`` or ``skip_rows`` is given
        or ``has_header`` is False, otherwise a copy with ``skip_lines``
        set to the result of ``detect_csv_skip_lines``.
    """
    if (
        "skip_lines" in kwargs
        or "skip_rows" in kwargs
        or kwargs.get("has_header", True) is False
    ):
        return kwargs
    skip_lines = detect_csv_skip_lines(
        source,
        separator=kwargs.get("separator", ","),
        quote_char=kwargs.get("quote_char", '"'),
    )
    return {**kwargs, "skip_lines": skip_lines}


def read_csv(source: str | Path, **kwargs: Any) -> pl.DataFrame:  # noqa: ANN401
    """Read a CSV, skipping the metadata lines above its header.

    Behaves as ``pl.read_csv`` except that ``skip_lines`` is detected with
    ``polars_helpers.utils.detect_csv_skip_lines``.

    Parameters
    ----------
    source : str | Path
        Path of the CSV file.
    **kwargs : Any
        Passed to ``pl.read_csv`` as they are. ``separator`` and
        ``quote_char`` are also used for the detection. When
        ``skip_lines`` or ``skip_rows`` is given, or ``has_header`` is
        False, nothing is detected.

    Returns
    -------
    pl.DataFrame
        The table.

    Raises
    ------
    ValueError
        If the header cannot be detected with confidence.
    """
    return pl.read_csv(source, **_with_skip_lines(source, kwargs))


def scan_csv(source: str | Path, **kwargs: Any) -> pl.LazyFrame:  # noqa: ANN401
    """Scan a CSV lazily, skipping the metadata lines above its header.

    Behaves as ``pl.scan_csv`` except that ``skip_lines`` is detected with
    ``polars_helpers.utils.detect_csv_skip_lines``, which reads only the
    first lines of the file.

    Parameters
    ----------
    source : str | Path
        Path of the CSV file.
    **kwargs : Any
        Passed to ``pl.scan_csv`` as they are. ``separator`` and
        ``quote_char`` are also used for the detection. When
        ``skip_lines`` or ``skip_rows`` is given, or ``has_header`` is
        False, nothing is detected.

    Returns
    -------
    pl.LazyFrame
        The table, not yet collected.

    Raises
    ------
    ValueError
        If the header cannot be detected with confidence.
    """
    return pl.scan_csv(source, **_with_skip_lines(source, kwargs))


def safe_write_csv(
    df: pl.DataFrame, file_path: str | Path, verbose: bool = False
) -> None:
    """Write a frame to CSV after dropping columns CSV cannot represent.

    Parameters
    ----------
    df : pl.DataFrame
        Frame to write. Nested-typed (List, Array, Struct) columns are
        dropped first.
    file_path : str | Path
        Destination CSV path.
    verbose : bool, default False
        Whether to print the names of the dropped columns and the
        destination path, which is intended for interactive notebook use.
    """
    drop_nested_type_columns(df, verbose).write_csv(file_path)
    if verbose:
        print(f"Written csv to {file_path}")
