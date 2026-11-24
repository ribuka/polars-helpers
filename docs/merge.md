# `merge`, `HOW`, and `VALIDATE`

Join two frames like `pandas.merge`: the keys of both sides become one set of
key columns, a `_merge` column tells where each row came from, and the row
order is fixed. `HOW` is the type of the join strategies it accepts, and
`VALIDATE` the type of the key relations its `validate` argument checks.

See the docstring for every argument.

## Try it

```python
import polars as pl
import polars_helpers as plh

left = pl.DataFrame({"id": [3, 1, 2], "name": ["c", "a", "b"]})
right = pl.DataFrame({"id": [1, 1, 4], "score": [10, 11, 40]})

out = plh.merge(left, right, on="id")
for row in out.to_dicts():
    print(row)
```

```text
{'id': 3, 'name': 'c', 'score': None, '_merge': 'left_only'}
{'id': 1, 'name': 'a', 'score': 10, '_merge': 'both'}
{'id': 1, 'name': 'a', 'score': 11, '_merge': 'both'}
{'id': 2, 'name': 'b', 'score': None, '_merge': 'left_only'}
```

The default is a left join. The rows follow the left frame, and the rows that
share one left row follow the right frame.

## Join strategies

`how` takes one of the values of `HOW`:

```python
import typing

print(typing.get_args(plh.HOW))
for how in ["inner", "right", "full"]:
    print(how, plh.merge(left, right, on="id", how=how).select("id", "_merge").rows())
```

```text
('left', 'right', 'full', 'inner')
inner [(1, 'both'), (1, 'both')]
right [(1, 'both'), (1, 'both'), (4, 'right_only')]
full [(3, 'left_only'), (1, 'both'), (1, 'both'), (2, 'left_only'), (4, 'right_only')]
```

- `"inner"` and `"left"` follow the left rows; `"right"` follows the right rows.
- `"full"` follows the left rows and then adds the right-only rows in the right
  order. This is the one difference from `pandas.merge`, which sorts the keys
  of an outer join.

## Keys with other names, fewer columns, no indicator

Use `left_on` / `right_on` when the keys are named differently. The key
columns are named after the left keys. `columns` takes only some columns of
`right` (its keys are always taken), and `indicator=None` leaves out the
`_merge` column, or `indicator="..."` gives it another name.

```python
prices = pl.DataFrame({"item_id": [1, 2], "price": [100, 200], "note": ["x", "y"]})
out = plh.merge(left, prices, left_on="id", right_on="item_id", columns=["price"], indicator=None)
print(out.columns)
print(out.rows())
```

```text
['id', 'name', 'price']
[(3, 'c', None), (1, 'a', 100), (2, 'b', 200)]
```

## Checking that keys are unique

`validate` checks the keys before merging, whatever `how` is, and raises a
`ValueError` that names the sides whose keys are not unique, how many key
values appear more than once, and the first five of them by key:

| `validate` | Keys that must be unique |
|---|---|
| `"m:m"` (default) | none; nothing is checked |
| `"1:m"` | left |
| `"m:1"` | right |
| `"1:1"` | left and right |

```python
print(typing.get_args(plh.VALIDATE))
try:
    plh.merge(left, right, on="id", validate="m:1")
except ValueError as error:
    print(error)
```

```text
('m:m', '1:m', 'm:1', '1:1')
validate='m:1': the keys of right are not unique
right keys ['id']: 1 key values appear more than once; the first 1 by key:
shape: (1, 2)
┌─────┬───────┐
│ id  ┆ count │
│ --- ┆ ---   │
│ i64 ┆ u32   │
╞═════╪═══════╡
│ 1   ┆ 2     │
└─────┴───────┘
```

Once the keys are unique, the merge goes on as without `validate`:

```python
latest = right.unique("id", keep="last", maintain_order=True)
print(plh.merge(left, latest, on="id", validate="1:1").rows())
```

```text
[(3, 'c', None, 'left_only'), (1, 'a', 11, 'both'), (2, 'b', None, 'left_only')]
```

- Rows with a null in any key column are not counted, as null keys match no
  row and never multiply rows.
- The check runs in the call, for `LazyFrame`s too: only the counts and the
  examples of the duplicated keys are collected, which reads the inputs once
  more. The returned `LazyFrame` stays unevaluated. `"m:m"` reads nothing.

## LazyFrames

Two `LazyFrame`s give a `LazyFrame`, with the same row order once collected.
Both sides must be of the same kind; a `DataFrame` and a `LazyFrame` raise a
`TypeError`.

```python
lazy = plh.merge(left.lazy(), right.lazy(), on="id", how="inner")
print(type(lazy).__name__)
print(lazy.collect().rows())
```

```text
LazyFrame
[(1, 'a', 10, 'both'), (1, 'a', 11, 'both')]
```

## Printing what happened

`verbose=True` prints the shape before and after. With an indicator it also
prints a table of the `_merge` counts.

```python
out = plh.merge(left, right, on="id", indicator=None, verbose=True)
```

```text
(3, 2) -> (4, 3)
```

## Things to know

- **Null keys never match.** Unlike `pandas.merge`, which matches missing keys
  with each other, a null key is `left_only` or `right_only`.
- **Duplicate keys multiply rows.** Each left row appears once for every
  matching right row, as `id` 1 does above, so the result can be longer than
  `left`. Pass `validate` to make sure it does not happen.
- **The indicator name must be free.** If a column named like `indicator`
  (`_merge` by default) already exists, a `ValueError` is raised.
- **Memory.** The join runs in memory when you collect. For frames too large
  for that, use [`merge_on_disk`](merge_on_disk.md), which gives the same
  result.
