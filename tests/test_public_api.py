"""Tests for the public API of ``polars_helpers``."""

import polars_helpers as plh


def test_public_api_exports_every_name() -> None:
    """Expose every name listed in ``__all__`` and nothing else by design."""
    assert sorted(plh.__all__) == [
        "HOW",
        "drop_all_null_columns",
        "drop_nested_type_columns",
        "get_columns",
        "get_numeric_columns",
        "get_schema",
        "get_shape",
        "get_unique_values",
        "get_value_counts",
        "merge",
        "safe_write_csv",
    ]
    for name in plh.__all__:
        assert hasattr(plh, name)


def test_natural_keys_is_not_top_level() -> None:
    """Keep ``natural_keys`` under ``polars_helpers.utils`` only."""
    assert not hasattr(plh, "natural_keys")
