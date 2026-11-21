# `get_numeric_columns`

Get the names of the numeric columns: integers, unsigned integers, floats,
and decimals.

## Try it

```python
from decimal import Decimal

import polars as pl
import polars_helpers as plh

df = pl.DataFrame(
    {
        "id": [1, 2],
        "name": ["a", "b"],
        "score": [0.5, 1.5],
        "price": [Decimal("1.20"), Decimal("3.40")],
        "ok": [True, False],
    }
)
print(plh.get_numeric_columns(df))
```

```text
['id', 'score', 'price']
```

## Only some columns

`columns` limits the candidates, and the result keeps their order. A name that
is not a column raises `pl.exceptions.ColumnNotFoundError`.

```python
print(plh.get_numeric_columns(df, columns=["score", "name", "id"]))
```

```text
['score', 'id']
```

A typical use is to pick the columns to aggregate:

```python
print(df.select(pl.col(plh.get_numeric_columns(df, columns=["id", "score"])).sum()).rows())
```

```text
[(3, 2.0)]
```

## Things to know

- `Boolean` is not numeric, nor are dates, times, and durations.
- A number stored as text (`"1"`) is a `String` column and is not included;
  cast it first.
