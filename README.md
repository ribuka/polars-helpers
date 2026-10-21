# polars-helpers

Small helpers for polars `DataFrame` / `LazyFrame`.

```python
import polars_helpers as plh

merged = plh.merge(left, right, on=["id"], how="left")
n_rows, n_cols = plh.get_shape(lazy_frame)  # does not collect the frame
```

- Helpers accept both `pl.DataFrame` and `pl.LazyFrame` where they can, and
  return the same kind they were given.
- A `LazyFrame` is never collected just to read its schema or shape.
- Nothing is printed unless `verbose=True` is passed.
- The package depends only on polars (1.18 or later).

## Install

```sh
uv add "polars-helpers @ git+https://github.com/ribuka/polars-helpers.git"
```

## API

See the docstrings for arguments and details.

| Name | Description |
|---|---|
| `merge` | Join two frames like `pandas.merge`, with coalesced keys, an indicator column, and the same row order. |
| `HOW` | The join strategies `merge` accepts. |
| `get_shape` | `(rows, columns)`. |
| `get_columns` | Column names. |
| `get_schema` | `pl.Schema`, optionally limited to some columns. |
| `get_numeric_columns` | Names of the numeric columns. |
| `get_unique_values` | Distinct non-null values of a column, sorted (strings naturally). |
| `get_value_counts` | Count of each distinct value of a column. |
| `read_csv` / `scan_csv` | `pl.read_csv` / `pl.scan_csv` that skip metadata lines above the header. |
| `drop_all_null_columns` | Drop columns that are (almost) entirely missing. |
| `drop_nested_type_columns` | Drop List, Array, and Struct columns. |
| `safe_write_csv` | Write a CSV after dropping nested-typed columns. |

`polars_helpers.utils` holds helpers that do not depend on polars:

| Name | Description |
|---|---|
| `detect_csv_skip_lines` | Number of metadata lines above the header of a CSV. |
| `natural_keys` | Sort key that orders embedded numbers numerically. |
