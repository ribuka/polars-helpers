# polars-helpers

Small helpers for polars `DataFrame` / `LazyFrame`.

```python
import polars_helpers as plh

merged = plh.merge(left, right, on=["id"], how="left")
n_rows, n_cols = plh.get_shape(lazy_frame)  # does not collect the frame
```

- Helpers accept both `pl.DataFrame` and `pl.LazyFrame` where they can, and
  return the same kind they were given. `merge_on_disk` is the exception: it
  accepts only `pl.LazyFrame` and always returns a `pl.LazyFrame`.
- A `LazyFrame` is never collected just to read its schema or shape.
- Nothing is printed unless `verbose=True` is passed.
- The package depends only on polars (2.0 or later).

## Install

```sh
uv add "polars-helpers @ git+https://github.com/ribuka/polars-helpers.git"
```

## API

See the docstrings for arguments and details.

| Name | Description |
|---|---|
| [`merge`](docs/merge.md) | Join two frames like `pandas.merge`, with coalesced keys, an indicator column, and a deterministic row order (pandas-compatible except for `how="full"`). |
| [`merge_on_disk`](docs/merge_on_disk.md) | Get the same result as `merge` with little memory by keeping the intermediate data on disk; the result is written to Parquet and returned as a `LazyFrame` that reads it. Accepts only `LazyFrame`. |
| [`HOW`](docs/merge.md) | The join strategies `merge` accepts. |
| [`scan_parquet_relaxed`](docs/scan_parquet_relaxed.md) | Stack Parquet files whose columns and types differ, like `diagonal_relaxed`, with little memory, by preparing their columns on disk; returns a `LazyFrame`. |
| [`get_shape`](docs/get_shape.md) | `(rows, columns)`. |
| [`get_columns`](docs/get_columns.md) | Column names. |
| [`get_schema`](docs/get_schema.md) | `pl.Schema`, optionally limited to some columns. |
| [`get_numeric_columns`](docs/get_numeric_columns.md) | Names of the numeric columns. |
| [`get_unique_values`](docs/get_unique_values.md) | Distinct non-null values of a column, sorted (strings naturally). |
| [`get_value_counts`](docs/get_value_counts.md) | Count of each distinct value of a column. |
| [`read_csv` / `scan_csv`](docs/read_csv.md) | `pl.read_csv` / `pl.scan_csv` that skip metadata lines above the header. |
| [`drop_all_null_columns`](docs/drop_all_null_columns.md) | Drop columns that are (almost) entirely missing. |
| [`drop_nested_type_columns`](docs/drop_nested_type_columns.md) | Drop List, Array, and Struct columns. |
| [`safe_write_csv`](docs/safe_write_csv.md) | Write a CSV after dropping nested-typed columns. |
