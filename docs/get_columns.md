# `get_columns`

Get the column names of a `DataFrame` or a `LazyFrame`, in their order,
without collecting the `LazyFrame`.

## Try it

```python
import polars as pl
import polars_helpers as plh

df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "score": [0.5, 1.5]})
print(plh.get_columns(df))

lf = df.lazy().with_columns((pl.col("score") * 2).alias("double"))
print(plh.get_columns(lf))
```

```text
['id', 'name', 'score']
['id', 'name', 'score', 'double']
```

## Things to know

- For a `LazyFrame`, the names come from `collect_schema()`, so no rows are
  read. The same call works for both kinds of frame.
- Resolving the schema of a `LazyFrame` can still read file metadata, for
  example the header of a CSV or the footer of a Parquet file.
- Anything other than a `DataFrame` or a `LazyFrame` raises a `TypeError`.
