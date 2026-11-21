# `scan_parquet_relaxed`

Stack Parquet files whose columns and types differ, without loading them
into memory, and get a plain `pl.LazyFrame` back.

The result is what `pl.concat(frames, how="diagonal_relaxed")` would give:
the columns of every file in the order they first appear, each with the type
that `vertical_relaxed` gives it, and nulls where a file lacks a column. Files
are stacked in the natural order of their names (`data_2` before `data_10`).

You do not manage any cache: where the files go, how long they stay, and
when they are deleted are handled for you. See the docstring for every
argument.

## Try it

Write two small Parquet files to a temporary folder and stack them.

```python
import tempfile
from pathlib import Path

import polars as pl
import polars_helpers as plh

folder = Path(tempfile.mkdtemp())
pl.DataFrame({"x": [1, 2], "label": ["a", "b"]}).write_parquet(folder / "data_10.parquet")
pl.DataFrame({"x": [0.5], "flag": [True]}).write_parquet(folder / "data_2.parquet")

lf = plh.scan_parquet_relaxed(folder)
print({name: str(dtype) for name, dtype in lf.collect_schema().items()})
for row in lf.collect().to_dicts():
    print(row)
```

```text
{'x': 'Float64', 'flag': 'Boolean', 'label': 'String'}
{'x': 0.5, 'flag': True, 'label': None}
{'x': 1.0, 'flag': None, 'label': 'a'}
{'x': 2.0, 'flag': None, 'label': 'b'}
```

`paths` can also be one file or a list of files. A folder is read without its
subfolders unless `recursive=True`.

## How it works, and what it costs

During the call, every column of every file is cast to its common type and
written, one column of one file at a time, to a Parquet file of its own
(a *part*). The returned frame reads the parts: one `scan_parquet` per column,
put side by side.

So the work is done **in the call**, not when you collect:

- With `columns=None` (the default), the first call writes **every column**
  of every file to disk. A later `select` on the returned frame reads fewer
  parts, but it does not undo that first preparation.
- Pass `columns` to prepare only the columns you need. The result has them in
  that order, and the types of the other columns are not even looked at.

```python
lf = plh.scan_parquet_relaxed(folder, columns=["label", "x"])
print(lf.collect().to_dicts())
```

```text
[{'label': None, 'x': 0.5}, {'label': 'a', 'x': 1.0}, {'label': 'b', 'x': 2.0}]
```

Preparing the parts needs little memory. `.collect()` of the whole result,
however, loads the result into memory like any collect; for a large result,
aggregate it or write it with a `sink_*` method:

```python
total = lf.select(pl.col("x").sum()).collect().item()
print(total)
out = Path(tempfile.mkdtemp()) / "result.parquet"
lf.filter(pl.col("x") > 0.5).sink_parquet(out)
print(pl.read_parquet(out).height)
```

```text
3.5
2
```

## Where the parts are kept

By default, in the user cache directory of the OS, so that every user of a
shared machine (a Remote Desktop server, for example) has their own:

- Windows: `%LOCALAPPDATA%\polars-helpers\cache\`
- macOS: `~/Library/Caches/polars-helpers/`
- Others: `$XDG_CACHE_HOME/polars-helpers/`, or `~/.cache/polars-helpers/`

The temporary directory of the OS is never used. Pass `directory=` to keep the
parts somewhere else; that directory is used as it is, and other files in it
are left alone.

```python
cache = Path(tempfile.mkdtemp()) / "my-cache"
lf = plh.scan_parquet_relaxed(folder, columns=["x"], directory=cache)
print(cache.is_dir())
```

```text
True
```

## How long the parts are kept

`cache_ttl_days` decides it:

- `1` (the default): parts are kept 1 day (24 hours) after their **last use
  by this function**. A later call, in this process or another one, reuses
  them while the files keep their size and modification time, and each reuse
  extends the time limit. Collecting the returned frame does not extend it.
- `n`: the same, with `n` days.
- `0`: nothing is kept for later calls; every call prepares new parts.
- `None`: parts are kept only for this process; later calls in it reuse them,
  and they are deleted when the process exits normally.

A file that changed (in size or modification time) is prepared again, and so
are the columns whose common type changed. A frame returned earlier still
reads the data as it was when it was returned:

```python
old = plh.scan_parquet_relaxed(folder, columns=["x"], cache_ttl_days=None)
pl.DataFrame({"x": [100]}).write_parquet(folder / "data_2.parquet")
new = plh.scan_parquet_relaxed(folder, columns=["x"], cache_ttl_days=None)
print(old.collect()["x"].to_list())
print(new.collect()["x"].to_list())
```

```text
[0.5, 1.0, 2.0]
[100, 1, 2]
```

## When the parts are deleted

- **In use**: the parts a returned frame reads are kept until the process
  exits, even past their time limit, and also for frames derived from it.
- **On normal exit**: parts kept with `None` are deleted, and so are expired
  parts this process used that no other process uses.
- **On each call**: the call deletes, in the directory it uses, the expired
  parts that no living process uses, and what killed processes left behind.

Nothing runs in the background, so expired parts stay on disk until the next
call on the same directory. Parts are never deleted to save space: the
directory grows with the files and columns you prepare, and a write fails
when the disk is full. When no process uses it, you can delete the directory
by hand.
