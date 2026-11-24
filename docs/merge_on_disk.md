# `merge_on_disk`

Get the same result as [`merge`](merge.md) with little memory: the merge is
computed with its intermediate data on disk, the result is written to Parquet
files, and a `pl.LazyFrame` that reads them is returned.

It accepts only `pl.LazyFrame` and always returns a `pl.LazyFrame`. The
arguments of `merge` (`on`, `left_on` / `right_on`, `columns`, `how`,
`indicator`, `validate`, `verbose`) work the same way. See the docstring for every
argument.

## Try it

```python
import tempfile
from pathlib import Path

import polars as pl
import polars_helpers as plh

folder = Path(tempfile.mkdtemp())
pl.DataFrame({"id": [3, 1, 2], "name": ["c", "a", "b"]}).write_parquet(folder / "left.parquet")
pl.DataFrame({"id": [1, 1, 4], "score": [10, 11, 40]}).write_parquet(folder / "right.parquet")

left = pl.scan_parquet(folder / "left.parquet")
right = pl.scan_parquet(folder / "right.parquet")

out = plh.merge_on_disk(left, right, on="id")
for row in out.collect().to_dicts():
    print(row)
```

```text
{'id': 3, 'name': 'c', 'score': None, '_merge': 'left_only'}
{'id': 1, 'name': 'a', 'score': 10, '_merge': 'both'}
{'id': 1, 'name': 'a', 'score': 11, '_merge': 'both'}
{'id': 2, 'name': 'b', 'score': None, '_merge': 'left_only'}
```

The rows, their order, the column names and the types are those of
`plh.merge(left, right, on="id").collect()`:

```python
same = plh.merge(left, right, on="id", how="full").collect()
print(plh.merge_on_disk(left, right, on="id", how="full").collect().equals(same))
```

```text
True
```

## The work is done in the call

The whole merge runs and its result is written **before** `merge_on_disk`
returns. A `select` or `filter` on the returned frame applies when the result
files are read; it does not make the merge cheaper. To merge less, narrow
`left` and `right` first, or take fewer right columns with `columns`:

```python
out = plh.merge_on_disk(left.filter(pl.col("id") > 1), right, on="id", columns=[], indicator=None)
print(out.collect().rows())
```

```text
[(3, 'c'), (2, 'b')]
```

## Where the result is written, and how long it stays

By default, in the user cache directory of the OS, the same directory as
[`scan_parquet_relaxed`](scan_parquet_relaxed.md) uses
(`%LOCALAPPDATA%\polars-helpers\cache\` on Windows,
`~/Library/Caches/polars-helpers/` on macOS, and `$XDG_CACHE_HOME/polars-helpers/`
or `~/.cache/polars-helpers/` elsewhere). The temporary directory of the OS is
never used. These files are deleted when the process exits normally, so use
the returned frame within the process that made it.

Pass `directory=` to write somewhere else. Each call writes files named
`merge-<id>-00000.parquet`, ... with an id of its own, and leaves the other
files of the directory alone. With `directory=`, the files are kept after the
process exits, unless `delete_on_exit=True`:

```python
results = folder / "results"
out = plh.merge_on_disk(left, right, on="id", directory=results)
names = [p.name for p in results.iterdir()]
print(all(name.startswith("merge-") and name.endswith(".parquet") for name in names))
print(pl.read_parquet(results).height)
```

```text
True
4
```

`delete_on_exit=False` keeps the result files of the default directory too.
Nothing else deletes them then, so delete them yourself when you are done.

## `buckets`

Both sides are split on disk into `buckets` parts (16 by default) by the hash
of their keys, and the parts are merged one at a time. `buckets=1` merges in
one go with the streaming engine; it is faster and uses less memory when
`right` is small, but keep the default when `right` is large.

```python
out = plh.merge_on_disk(left, right, on="id", buckets=1, indicator=None)
print(out.collect().rows())
```

```text
[(3, 'c', None), (1, 'a', 10), (1, 'a', 11), (2, 'b', None)]
```

## Things to know

- **Disk.** With `buckets` of 2 or more, the call holds, at its peak, a copy
  of both sides and about twice the result on disk, as Parquet. While a side
  is split into buckets, a copy of that side is also on disk; it is deleted
  once the side is split. A write fails when the disk is full. The working files are deleted before the call
  returns, and everything the call wrote is deleted when it fails.
- **Memory.** Collecting the whole returned frame loads the result into
  memory like any collect. For a large result, aggregate it, or write it with
  a `sink_*` method. Inside the call, with `buckets` of 2 or more, the merged
  rows are put back in order about 1 GiB at a time (about twice that at the
  peak, while sorting). The size of a row is measured on the first 10,000
  merged rows, so data whose later rows are much larger can go beyond it, and
  the rows one input row gives are always handled together, so a row that
  matches many rows takes the memory of all its matches at once.
- **Rows.** With `buckets` of 2 or more, each side can have at most
  `2**32 - 1` rows, because the rows are numbered with `pl.get_index_type()`,
  which is `UInt32` in the default polars runtime. A larger side fails inside
  polars, sometimes with a panic that is not a subclass of `Exception`.
  `buckets=1` has no such limit, as it numbers no rows. With polars installed
  as `polars[rt64]`, the row numbers are `UInt64` and the limit does not
  apply.
- **LazyFrames only.** Pass `df.lazy()` for a `DataFrame`; anything else
  raises a `TypeError`.
- **Same rules as `merge`.** Null keys never match, duplicate keys multiply
  rows, and the indicator name must not be a column already.
- **`validate`.** It raises the same `ValueError` as with `merge`. With
  `buckets=1`, both sides are checked with the streaming engine before
  anything is written, and the memory grows with the number of distinct keys.
  With `buckets` of 2 or more, the keys are checked a bucket at a time, once
  both sides are split into buckets and before any bucket is merged: the
  memory is that of one bucket, but the error comes only after the split
  (about 4 s after the call with 30 million rows on each side). Everything
  the call wrote is deleted before the error is raised.
