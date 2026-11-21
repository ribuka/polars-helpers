# `read_csv` and `scan_csv`

`pl.read_csv` and `pl.scan_csv` that find the header themselves: lines of
metadata above the header (a title, an export date, ...) are skipped. Every
keyword argument is passed to polars as it is.

`read_csv` returns a `DataFrame`; `scan_csv` returns a `LazyFrame` and reads
only the first lines of the file to find the header.

## Try it

Write a CSV with two metadata lines above its header, and read it.

```python
import tempfile
from pathlib import Path

import polars as pl
import polars_helpers as plh

folder = Path(tempfile.mkdtemp())
path = folder / "sales.csv"
path.write_text(
    "Sales report\n"
    "Exported: 2024-04-01\n"
    "\n"
    "id,name,amount\n"
    "1,apple,120\n"
    "2,banana,80\n"
    "3,cherry,300\n",
    encoding="utf-8",
)

df = plh.read_csv(path)
print(df.columns)
print(df.rows())

lf = plh.scan_csv(path)
print(lf.filter(pl.col("amount") > 100).collect().rows())
```

```text
['id', 'name', 'amount']
[(1, 'apple', 120), (2, 'banana', 80), (3, 'cherry', 300)]
[(1, 'apple', 120), (3, 'cherry', 300)]
```

`pl.read_csv(path)` would take `Sales report` as the header and fail on the
rows below it.

## Other formats

`separator`, `quote_char`, and `encoding` are passed to polars and also used
to find the header:

```python
path = folder / "semicolon.csv"
path.write_text("# exported by a tool\nid;city\n1;Tokyo\n2;Osaka\n", encoding="utf-8")
print(plh.read_csv(path, separator=";").rows())
```

```text
[(1, 'Tokyo'), (2, 'Osaka')]
```

When you pass `skip_lines` or `skip_rows`, or `has_header=False`, nothing is
detected, and the call is the same as the polars one:

```python
print(plh.read_csv(path, separator=";", skip_lines=1).columns)
```

```text
['id', 'city']
```

## When the header cannot be found

If no line starts a table with confidence, a `ValueError` is raised instead of
guessing. Pass `skip_lines` yourself then.

```python
path = folder / "short.csv"
path.write_text("id,name\n1,apple\n", encoding="utf-8")
try:
    plh.read_csv(path)
except ValueError:
    print("not detected")
print(plh.read_csv(path, skip_lines=0).rows())
```

```text
not detected
[(1, 'apple')]
```

## Things to know

- The header is looked for in the first 200 lines. It needs at least two data
  rows below it, all with the same number of cells.
- A metadata line with as many cells as the table must be separated from it
  by an empty line. Otherwise it cannot be told apart from the header, and
  either it is taken as the header (when every column is text) or a
  `ValueError` is raised.
- The detection reads the file with Python, so `source` must be a path to a
  local file.
- See `polars_helpers.utils.detect_csv_skip_lines` for the exact rules.
