# `safe_write_csv`

Write a `DataFrame` to CSV after dropping the `List`, `Array`, and `Struct`
columns, which `write_csv` would otherwise refuse.

## Try it

```python
import tempfile
from pathlib import Path

import polars as pl
import polars_helpers as plh

df = pl.DataFrame({"id": [1, 2], "tags": [["a", "b"], ["c"]], "name": ["first", "second"]})

path = Path(tempfile.mkdtemp()) / "out.csv"
plh.safe_write_csv(df, path)
print(path.read_text(encoding="utf-8"))
```

```text
id,name
1,first
2,second
```

`df.write_csv(path)` would raise an error because of the `tags` column.

## Seeing what was dropped

`verbose=True` prints the dropped columns and then the destination, as
`drop_cols = ['tags']` and `Written csv to <file_path>`.

```python
plh.safe_write_csv(df, path, verbose=True)
```

## Things to know

- **The data in nested columns is lost** without a warning, unless
  `verbose=True`. To keep it, flatten those columns first (see
  [`drop_nested_type_columns`](drop_nested_type_columns.md)).
- **Only `DataFrame`.** For a `LazyFrame`, use
  `plh.drop_nested_type_columns(lf).sink_csv(path)`, which writes without
  loading the whole frame into memory.
- An existing file at `file_path` is overwritten.
- Other `write_csv` options (separator, date format, ...) are not available;
  call `plh.drop_nested_type_columns(df).write_csv(path, ...)` when you need
  them.
