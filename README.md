# polars-helpers

Small helpers for polars `DataFrame` / `LazyFrame`.

```python
import polars_helpers as plh

merged = plh.merge(left, right, on=["id"], how="left")
n_rows, n_cols = plh.get_shape(lazy_frame)  # does not collect the frame
```

- Every helper that takes a frame accepts both `pl.DataFrame` and
  `pl.LazyFrame` (except `get_numeric_columns` and `safe_write_csv`, which take a
  `pl.DataFrame`), and returns the same kind it was given; the type hints say
  so too.
- Nothing is printed unless `verbose=True` is passed.
- The package depends only on polars (1.18 or later).

## Install

```sh
uv add "polars-helpers @ git+https://github.com/ribuka/polars-helpers.git"
```

## API

### Merge

| Name | Description |
|---|---|
| `merge(left, right, columns=None, on=None, left_on=None, right_on=None, how="left", indicator="_merge", verbose=False)` | Join two frames of the same kind like `pandas.merge`: the keys of both sides are coalesced into one set of key columns (named after the left keys), and an indicator column holds `"both"`, `"left_only"`, or `"right_only"` (`indicator=None` omits it). Keys are given as `on`, or as `left_on` and `right_on` (a name or a list). `columns` limits the columns taken from `right`; its keys are always taken. The row order is kept as with `pandas.merge`, for `pl.LazyFrame` too: `"left"` and `"inner"` follow the left rows, `"right"` follows the right rows, and the rows sharing one row of that side follow the other side; `"full"` follows the left rows and then the right-only rows (pandas sorts the keys instead). With `verbose`, prints the shape change and the indicator counts. |
| `HOW` | `Literal["left", "right", "full", "inner"]`, the join strategies `merge` accepts. |

### Information

| Name | Description |
|---|---|
| `get_shape(df)` | `(rows, columns)`. |
| `get_columns(df)` | Column names. |
| `get_schema(df, columns=None)` | `pl.Schema`, optionally limited to `columns`. |
| `get_numeric_columns(df, columns=None)` | Names of the numeric (integer, unsigned integer, float, or decimal) columns (among `columns`). |
| `get_unique_values(df, column, sort=True)` | Distinct non-null values of one column, sorted by value; String columns are sorted naturally (`item-2` before `item-10`). |
| `get_value_counts(df, column, sort=True)` | Count of each distinct value of one column. |

### Input

| Name | Description |
|---|---|
| `read_csv(source, **kwargs)` | `pl.read_csv(source, **kwargs)` with `skip_lines` detected by `detect_csv_skip_lines`, so that metadata lines above the header are skipped. `source` is a path; the `separator`, `quote_char`, and `encoding` of `kwargs` are also used for the detection. Nothing is detected when `skip_lines` or `skip_rows` is given or `has_header=False`. Raises `ValueError` when the header cannot be detected with confidence. |
| `scan_csv(source, **kwargs)` | The same for `pl.scan_csv`, returning a `pl.LazyFrame`; only the first lines are read for the detection. |

### Cleaning and output

| Name | Description |
|---|---|
| `drop_all_null_columns(df, include_nan_missing=True, threshold=0.99)` | Drop columns whose missing ratio (null, and NaN for float columns) is above `threshold` (`ValueError` unless `0.0 <= threshold <= 1.0`). |
| `drop_nested_type_columns(df, verbose=False)` | Drop nested-typed (List, Array, Struct) columns. With `verbose`, prints their names. |
| `safe_write_csv(df, file_path, verbose=False)` | Write a CSV after dropping nested-typed columns. With `verbose`, prints the dropped names and the path. |

### `polars_helpers.utils`

Helpers that do not depend on polars. They are not exported from the top level.

| Name | Description |
|---|---|
| `detect_csv_skip_lines(source, *, separator=",", quote_char='"', encoding="utf-8-sig", sample_lines=200)` | Number of physical lines before the header of a CSV that starts with metadata, for `skip_lines=`. The first non-empty line from which the rest of the first `sample_lines` lines reads as a table is taken: the same number (two or more) of cells in every record, no empty line, at least two data rows, every column either all numeric or free of numbers (ignoring missing tokens such as `NA`), and the header not numeric in at least one numeric column. Raises `ValueError` for an empty file, when no line qualifies, and when the line just above the header is not empty and has as many cells (metadata of the same width must be followed by an empty line; a two-line header is not supported). An all-text table preceded by same-width metadata without an empty line is read with that metadata as its header. |
| `natural_keys(text)` | Sort key that orders embedded numbers numerically: `sorted(values, key=natural_keys)`. |

## Development

```sh
uv sync
uv run -m ruff check .
uv run -m pyright
uv run -m pytest
```

CI runs lint, the type check, and the tests on every pull request to `main`,
and runs the tests again with the lowest polars that `pyproject.toml` allows.
