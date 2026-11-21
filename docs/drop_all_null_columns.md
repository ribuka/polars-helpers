# `drop_all_null_columns`

Drop the columns that are (almost) entirely missing. A column is kept when its
share of missing values is at most `threshold` (`0.99` by default).

## Try it

```python
import polars as pl
import polars_helpers as plh

df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "empty": [None, None, None, None],
        "nan": [float("nan"), None, float("nan"), float("nan")],
        "half": [1.0, None, None, 4.0],
    }
)
print(plh.drop_all_null_columns(df).columns)
```

```text
['id', 'half']
```

By default NaN counts as missing in float columns, so `nan` is dropped too.

## Choosing what is missing, and how much

`include_nan_missing=False` counts only nulls. `threshold` is the largest
share of missing values a kept column may have, from `0.0` to `1.0`.

```python
print(plh.drop_all_null_columns(df, include_nan_missing=False).columns)
print(plh.drop_all_null_columns(df, threshold=0.5).columns)
print(plh.drop_all_null_columns(df, threshold=0.0).columns)
```

```text
['id', 'nan', 'half']
['id', 'half']
['id']
```

`half` is missing in half of its rows: kept with `threshold=0.5`, dropped with
`threshold=0.0`.

## LazyFrames

A `LazyFrame` gives a `LazyFrame`. To decide which columns to drop, the counts
of rows and of missing values are computed: the data is read once, but only
those counts are loaded into memory.

```python
lf = plh.drop_all_null_columns(df.lazy())
print(type(lf).__name__, lf.collect_schema().names())
```

```text
LazyFrame ['id', 'half']
```

## Things to know

- **The default keeps a little.** With `threshold=0.99`, a column with one
  value in 100 rows is kept, but one value in 101 rows or more is dropped.
  Use `threshold=1.0` to drop nothing, or a lower value to drop more.
- **Empty frames are kept as they are.** With no rows, no column is dropped.
- **A `LazyFrame` is read when you call it,** to count the missing values,
  and again when you collect the result.
- `threshold` outside `0.0` to `1.0` raises a `ValueError`.
