# `drop_nested_type_columns`

Drop the `List`, `Array`, and `Struct` columns, which flat formats such as
CSV cannot hold.

## Try it

```python
import polars as pl
import polars_helpers as plh

df = pl.DataFrame(
    {
        "id": [1, 2],
        "tags": [["a", "b"], ["c"]],
        "point": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        "name": ["first", "second"],
    }
)
print(plh.drop_nested_type_columns(df).columns)
```

```text
['id', 'name']
```

## Seeing what was dropped

`verbose=True` prints the names of the dropped columns.

```python
lf = plh.drop_nested_type_columns(df.lazy(), verbose=True)
print(type(lf).__name__, lf.collect_schema().names())
```

```text
drop_cols = ['tags', 'point']
LazyFrame ['id', 'name']
```

A `LazyFrame` gives a `LazyFrame`; only its schema is read.

## Things to know

- **The data in those columns is lost.** To keep it, turn the columns into
  flat ones first, for example with `pl.col("tags").list.join(",")` or
  `unnest("point")`.
- Only the type decides: a `List` column is dropped even if every value is
  null or empty.
- [`safe_write_csv`](safe_write_csv.md) calls this before writing a CSV.
