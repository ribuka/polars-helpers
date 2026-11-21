# `get_value_counts`

Count each distinct value of one column, nulls included, and get a
`DataFrame` sorted by descending count.

## Try it

```python
import polars as pl
import polars_helpers as plh

df = pl.DataFrame({"color": ["red", "blue", "red", None, "green", "blue", "red"]})
for row in plh.get_value_counts(df, "color").to_dicts():
    print(row)
```

```text
{'color': 'red', 'count': 3}
{'color': 'blue', 'count': 2}
{'color': None, 'count': 1}
{'color': 'green', 'count': 1}
```

Values with the same count are sorted by value, with nulls first, so the order
is the same on every run.

## LazyFrames

A `LazyFrame` is aggregated before it is collected: only one row per distinct
value is loaded, and a `DataFrame` is returned.

```python
counts = plh.get_value_counts(df.lazy(), "color")
print(type(counts).__name__, counts.height)
```

```text
DataFrame 4
```

## Without sorting

`sort=False` skips the sort, and the rows are then in no particular order,
which can change from one run to the next. Sort the result yourself when you
need an order:

```python
counts = plh.get_value_counts(df, "color", sort=False)
print(counts.sort("color", nulls_last=True).rows())
```

```text
[('blue', 2), ('green', 1), ('red', 3), (None, 1)]
```

## Things to know

- The count column is named `count`.
- The type of `count` is the index type of your polars build (`UInt32`, or
  `UInt64` with the 64-bit runtime); cast it if you need a fixed type.
- The result has one row per distinct value, so a column with very many
  distinct values gives a long result.
