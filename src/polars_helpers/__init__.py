"""Small helpers for polars DataFrames and LazyFrames.

Use it as ``import polars_helpers as plh``.
"""

from .cleaning import (
    drop_all_null_columns,
    drop_nested_type_columns,
)
from .info import (
    get_columns,
    get_numeric_columns,
    get_schema,
    get_shape,
    get_unique_values,
    get_value_counts,
)
from .io import read_csv, safe_write_csv, scan_csv
from .merge import HOW, merge
from .merge_on_disk import merge_on_disk
from .parquet import scan_parquet_relaxed

__all__ = [
    "HOW",
    "drop_all_null_columns",
    "drop_nested_type_columns",
    "get_columns",
    "get_numeric_columns",
    "get_schema",
    "get_shape",
    "get_unique_values",
    "get_value_counts",
    "merge",
    "merge_on_disk",
    "read_csv",
    "safe_write_csv",
    "scan_csv",
    "scan_parquet_relaxed",
]
