# `get_shape`

Get `(rows, columns)` of a `DataFrame` or a `LazyFrame`, without collecting
the `LazyFrame`.

## Try it

```python
import tempfile
from pathlib import Path

import polars as pl
import polars_helpers as plh

df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
print(plh.get_shape(df))

folder = Path(tempfile.mkdtemp())
df.write_parquet(folder / "data.parquet")
lf = pl.scan_parquet(folder / "data.parquet").filter(pl.col("id") > 1)
print(plh.get_shape(lf))
```

```text
(3, 2)
(2, 2)
```

For a `DataFrame` it is the same as `df.shape`. A `LazyFrame` has no `shape`,
and `lf.collect().shape` would load the whole frame into memory.

## Things to know

- **The rows are still counted.** For a `LazyFrame`, only a `pl.len()` query
  is collected, so the frame is not loaded into memory, but the query still
  runs: the files are read and the filters and joins of the plan are
  computed. Parquet files without filters can be counted from their metadata;
  a CSV is read through.
- The column count comes from the schema, which does not read the data.
- Anything other than a `DataFrame` or a `LazyFrame` raises a `TypeError`.
