# `get_unique_values`

Get the distinct non-null values of one column as a Python list, sorted by
default. Strings are sorted naturally: `item2` comes before `item10`.

## Try it

```python
import polars as pl
import polars_helpers as plh

df = pl.DataFrame(
    {
        "item": ["item10", "item2", None, "item1", "item2"],
        "size": [3, 1, 2, None, 1],
    }
)
print(plh.get_unique_values(df, "item"))
print(plh.get_unique_values(df.lazy(), "size"))
```

```text
['item1', 'item2', 'item10']
[1, 2, 3]
```

`sorted(df["item"].drop_nulls().unique())` would give
`['item1', 'item10', 'item2']` instead.

## Without sorting

`sort=False` skips the sort. The values are then in no particular order, which
can change from one run to the next, so use it only when the order does not
matter:

```python
print(set(plh.get_unique_values(df, "item", sort=False)) == {"item1", "item2", "item10"})
```

```text
True
```

## Things to know

- **Nulls are dropped.** Use [`get_value_counts`](get_value_counts.md) to see
  how many there are.
- **Memory.** The distinct values are collected and turned into Python
  objects. For a column with millions of distinct values, aggregate in polars
  instead.
- Only `String` columns are sorted naturally; other columns, `Categorical`
  included, are sorted by polars.
- For a `LazyFrame`, only the distinct values of that column are collected.
