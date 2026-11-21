# `get_schema`

Get the `pl.Schema` of a `DataFrame` or a `LazyFrame`, optionally only for
some columns and in their order, without collecting the `LazyFrame`.

## Try it

```python
import polars as pl
import polars_helpers as plh

df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "score": [0.5, 1.5]})
schema = plh.get_schema(df.lazy())
print({name: str(dtype) for name, dtype in schema.items()})
```

```text
{'id': 'Int64', 'name': 'String', 'score': 'Float64'}
```

## Only some columns

`columns` limits the schema to those columns, in the order you give.
A name that is not a column raises `pl.exceptions.ColumnNotFoundError`.

```python
schema = plh.get_schema(df, columns=["score", "id"])
print({name: str(dtype) for name, dtype in schema.items()})

try:
    plh.get_schema(df, columns=["id", "missing"])
except pl.exceptions.ColumnNotFoundError as error:
    print(type(error).__name__)
```

```text
{'score': 'Float64', 'id': 'Int64'}
ColumnNotFoundError
```

## Things to know

- `columns=None` and `columns=[]` both give the whole schema.
- The result is a `pl.Schema`, which behaves like a dict from names to types
  (`schema["id"]`, `schema.names()`, `schema.items()`).
- Anything other than a `DataFrame` or a `LazyFrame` raises a `TypeError`.
