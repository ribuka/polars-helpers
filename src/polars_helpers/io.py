"""Write polars frames to files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .cleaning import drop_nested_type_columns


def safe_write_csv(
    df: pl.DataFrame, file_path: str | Path
) -> None:
    """Write a frame to CSV after dropping columns CSV cannot represent.

    Parameters
    ----------
    df : pl.DataFrame
        Frame to write. Nested-typed (List, Array, Struct) columns are
        dropped first.
    file_path : str | Path
        Destination CSV path.
    """
    df = drop_nested_type_columns(df)  # ty:ignore[invalid-assignment]
    df.write_csv(file_path)
    print(f"Written csv to {file_path}")
