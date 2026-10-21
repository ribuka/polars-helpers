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
from .io import safe_write_csv
from .merge import HOW, merge

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
    "safe_write_csv",
]
