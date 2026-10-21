"""Helpers that do not depend on polars."""

from .csv_header import detect_csv_skip_lines
from .natural_keys import natural_keys

__all__ = ["detect_csv_skip_lines", "natural_keys"]
