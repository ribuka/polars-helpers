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

### CSV input

| Name | Description |
|---|---|
| `read_csv(source, *, separator=",", encoding="utf-8-sig", **kwargs)` | Read a CSV file whose table may follow metadata lines (equipment info, conditions, ...): the header line is found with `detect_csv_skip_lines` and passed to `pl.read_csv` as `skip_lines`, with `**kwargs`. `encoding` is used only for the detection (polars reads UTF-8 only). `skip_lines`, `skip_rows`, or `has_header=False` raise `TypeError`; use `pl.read_csv` to give them by hand. |
| `scan_csv(source, *, separator=",", encoding="utf-8-sig", **kwargs)` | The same for `pl.scan_csv`, returning a `pl.LazyFrame`; only the first lines are read for the detection. |

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
| `detect_csv_skip_lines(source, *, separator=",", encoding="utf-8-sig", sample_lines=200)` | Number of physical lines before the header line of a CSV file that may start with metadata, to pass as `skip_lines=` (not `skip_rows=`). Only the first `sample_lines` lines are read: the field count of the last line is the table width, and the header is the lowest line of that width whose cells contradict the number or date/time columns below it (or the top of that run of lines if there is no such cue). Raises `ValueError`, showing the width and the candidate lines, instead of guessing: for an empty file, fewer than 2 columns, fewer than 2 data lines, a two-line header (names and units), or data-like lines of the table width above the header (also before an empty line or a field-count change, so a break inside the data never drops the rows before it), or a quoted cell spanning lines at or after the header. Metadata with the table's field count and no empty line before the header is told apart only by type cues, so with all-text columns its first line is taken as the header (and before an empty line it raises). |
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
