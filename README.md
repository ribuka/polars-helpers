# polars-helpers

Small helpers for polars `DataFrame` / `LazyFrame`.

```python
import polars_helpers as plh

merged = plh.merge(left, right, on=["id"], how="left")
n_rows, n_cols = plh.get_shape(lazy_frame)  # does not collect the frame
```

- Every helper that takes a frame accepts both `pl.DataFrame` and
  `pl.LazyFrame` (except `get_numeric_args` and `safe_write_csv`, which take a
  `pl.DataFrame`), and returns the same kind it was given.
- The package depends only on polars (1.1 or later).

## Install

```sh
uv add "polars-helpers @ git+https://github.com/ribuka/polars-helpers.git"
```

## API

### Merge

| Name | Description |
|---|---|
| `merge(left, right, columns=None, on=None, left_on=None, right_on=None, how="left", indicator="_merge", verbose=False)` | Join two frames of the same kind like `pandas.merge`: the keys of both sides are coalesced into one set of key columns (named after the left keys), and an indicator column holds `"both"`, `"left_only"`, or `"right_only"` (`indicator=None` omits it). Keys are given as `on`, or as `left_on` and `right_on` (a name or a list). `columns` limits the columns taken from `right`; its keys are always taken. With `verbose`, prints the shape change and the indicator counts. |
| `HOW` | `Literal["left", "right", "full", "inner"]`, the join strategies `merge` accepts. |

### Information

| Name | Description |
|---|---|
| `get_shape(df)` | `(rows, columns)`. |
| `get_columns(df)` | Column names. |
| `get_schema(df, args=None)` | Schema, optionally limited to `args`. |
| `get_numeric_args(df, args=None)` | Names of the numeric columns (among `args`). |
| `get_unique_values(df, alias, sort=True)` | Distinct non-null values of one column, sorted naturally (`item-2` before `item-10`). |
| `get_value_counts(df, alias, sort=True)` | Count of each distinct value of one column. |

### Cleaning and output

| Name | Description |
|---|---|
| `drop_all_null_columns(frame, include_nan_missing=True, threshold=0.99)` | Drop columns whose missing ratio (null, and NaN for float columns) is above `threshold`. |
| `validate_missing_ratio_threshold(threshold)` | Raise `ValueError` unless `0.0 <= threshold <= 1.0`. |
| `drop_list_type_columns(df)` | Drop List-typed columns and print their names. |
| `safe_write_csv(df, file_path)` | Write a CSV after dropping List-typed columns. |

### `polars_helpers.utils`

Helpers that do not depend on polars. They are not exported from the top level.

| Name | Description |
|---|---|
| `natural_keys(text)` | Sort key that orders embedded numbers numerically: `sorted(values, key=natural_keys)`. |

## Development

```sh
uv sync
uv run -m ruff check .
uv run -m pytest
```

CI runs lint and the tests on every pull request to `main`, and runs the tests
again with the lowest polars that `pyproject.toml` allows.
