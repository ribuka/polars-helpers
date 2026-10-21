"""Read and write polars frames as files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .cleaning import drop_nested_type_columns
from .utils.csv_skip_lines import detect_csv_skip_lines


def _detect_skip_lines(
    source: str | Path, separator: str, encoding: str, kwargs: dict[str, Any]
) -> int:
    """Reject arguments that contradict header detection, then detect it.

    Parameters
    ----------
    source : str | Path
        CSV file path.
    separator : str
        Field separator.
    encoding : str
        Text encoding used for the detection.
    kwargs : dict[str, Any]
        Keyword arguments meant for the polars reader.

    Returns
    -------
    int
        Number of physical lines before the header line.

    Raises
    ------
    TypeError
        If ``kwargs`` has ``skip_lines``, ``skip_rows``, or
        ``has_header=False``.
    """
    for name in ("skip_lines", "skip_rows"):
        if name in kwargs:
            raise TypeError(
                f"{name} contradicts header detection; "
                "use polars directly to give it by hand."
            )
    if kwargs.get("has_header", True) is False:
        raise TypeError(
            "has_header=False contradicts header detection; "
            "use polars directly to read a CSV without a header."
        )
    return detect_csv_skip_lines(
        source,
        separator=separator,
        quote_char=kwargs.get("quote_char", '"'),
        encoding=encoding,
    )


def read_csv(
    source: str | Path,
    *,
    separator: str = ",",
    encoding: str = "utf-8-sig",
    **kwargs: Any,
) -> pl.DataFrame:
    """Read a CSV file after skipping the metadata lines before its header.

    The header line is found with
    ``polars_helpers.utils.detect_csv_skip_lines``.

    Parameters
    ----------
    source : str | Path
        CSV file path. polars reads only UTF-8 text.
    separator : str, default ","
        Field separator.
    encoding : str, default "utf-8-sig"
        Text encoding used for the header detection only.
    **kwargs : Any
        Passed to ``pl.read_csv``.

    Returns
    -------
    pl.DataFrame
        The table that starts at the detected header line.

    Raises
    ------
    TypeError
        If ``kwargs`` has ``skip_lines``, ``skip_rows``, or
        ``has_header=False``.
    ValueError
        If the header line cannot be detected.
    """
    skip_lines = _detect_skip_lines(source, separator, encoding, kwargs)
    return pl.read_csv(source, skip_lines=skip_lines, separator=separator, **kwargs)


def scan_csv(
    source: str | Path,
    *,
    separator: str = ",",
    encoding: str = "utf-8-sig",
    **kwargs: Any,
) -> pl.LazyFrame:
    """Lazily scan a CSV file after skipping the metadata lines before its header.

    Only the first lines are read to detect the header line, with
    ``polars_helpers.utils.detect_csv_skip_lines``.

    Parameters
    ----------
    source : str | Path
        CSV file path. polars reads only UTF-8 text.
    separator : str, default ","
        Field separator.
    encoding : str, default "utf-8-sig"
        Text encoding used for the header detection only.
    **kwargs : Any
        Passed to ``pl.scan_csv``.

    Returns
    -------
    pl.LazyFrame
        The table that starts at the detected header line.

    Raises
    ------
    TypeError
        If ``kwargs`` has ``skip_lines``, ``skip_rows``, or
        ``has_header=False``.
    ValueError
        If the header line cannot be detected.
    """
    skip_lines = _detect_skip_lines(source, separator, encoding, kwargs)
    return pl.scan_csv(source, skip_lines=skip_lines, separator=separator, **kwargs)


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
