"""Detection of the header line of a CSV preceded by metadata lines."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none"})
_MIN_DATA_ROWS = 2


def _is_blank(line: str) -> bool:
    """Tell whether a physical line holds nothing but whitespace.

    Parameters
    ----------
    line : str
        Physical line, with or without its line ending.

    Returns
    -------
    bool
        True when the line is empty after stripping whitespace.
    """
    return not line.strip()


def _is_number(cell: str) -> bool:
    """Tell whether a cell is a plain decimal number.

    Parameters
    ----------
    cell : str
        Cell text.

    Returns
    -------
    bool
        True when the stripped cell is an integer or a decimal, optionally
        signed and with an exponent.
    """
    return _NUMBER.fullmatch(cell.strip()) is not None


def _is_missing(cell: str) -> bool:
    """Tell whether a cell is a missing-value token.

    Parameters
    ----------
    cell : str
        Cell text.

    Returns
    -------
    bool
        True for an empty cell and for ``NA``, ``N/A``, ``NaN``, ``null``,
        and ``None`` in any case.
    """
    return cell.strip().lower() in _MISSING_TOKENS


def _parse(
    lines: Sequence[str], separator: str, quote_char: str | None
) -> list[list[str]] | None:
    """Parse physical lines into CSV records.

    Parameters
    ----------
    lines : Sequence[str]
        Physical lines, each keeping its line ending so that quoted cells
        spanning lines are joined.
    separator : str
        Field separator.
    quote_char : str | None
        Quote character, or None to treat quotes as ordinary characters.

    Returns
    -------
    list[list[str]] | None
        Records, where an empty line gives an empty record, or None when
        the lines cannot be parsed as CSV. Cells are not limited by
        ``csv.field_size_limit``, which is restored afterwards.
    """
    if quote_char is None:
        reader = csv.reader(lines, delimiter=separator, quoting=csv.QUOTE_NONE)
    else:
        reader = csv.reader(lines, delimiter=separator, quotechar=quote_char)
    limit = csv.field_size_limit()
    csv.field_size_limit(max(limit, sum(len(line) for line in lines)))
    try:
        return list(reader)
    except csv.Error:
        return None
    finally:
        csv.field_size_limit(limit)


def _forms_table(records: list[list[str]]) -> bool:
    """Tell whether records read as a header followed by a consistent table.

    Parameters
    ----------
    records : list[list[str]]
        Records starting at the candidate header.

    Returns
    -------
    bool
        True when every record has the same number (two or more) of
        cells, at least ``_MIN_DATA_ROWS`` data records follow the header,
        every column is either all numeric or free of numbers (ignoring
        missing-value tokens), and the header is not numeric in at least
        one numeric column (when there is one).
    """
    if not records:
        return False
    width = len(records[0])
    if width < 2 or len(records) - 1 < _MIN_DATA_ROWS:
        return False
    if any(len(record) != width for record in records):
        return False

    header, data = records[0], records[1:]
    numeric_columns = []
    for index in range(width):
        values = [row[index] for row in data if not _is_missing(row[index])]
        n_numbers = sum(_is_number(value) for value in values)
        if 0 < n_numbers < len(values):
            return False
        if n_numbers:
            numeric_columns.append(index)
    return not numeric_columns or any(
        not _is_number(header[index]) for index in numeric_columns
    )


def detect_csv_skip_lines(
    source: str | Path,
    *,
    separator: str = ",",
    quote_char: str | None = '"',
    encoding: str = "utf-8-sig",
    sample_lines: int = 200,
) -> int:
    """Detect how many metadata lines precede the header of a CSV.

    Each non-empty physical line, from the top, is tried as the header,
    and the first one from which the rest of the sample reads as a table
    is taken. The rest reads as a table when every record has the same
    number ``W`` (two or more) of cells with no empty line in between, at
    least two data records follow the header, every column is either all
    numeric or free of numbers (ignoring missing-value tokens such as
    ``NA``), and, when there is a numeric column, the header is not
    numeric in at least one of them. Lines above a candidate are not
    parsed, and a candidate whose rest cannot be parsed is skipped, so an
    unclosed quote in the metadata does no harm.

    Parameters
    ----------
    source : str | Path
        Path of the CSV file.
    separator : str, default ","
        Field separator.
    quote_char : str | None, default '"'
        Quote character, or None to treat quotes as ordinary characters.
    encoding : str, default "utf-8-sig"
        Text encoding. Undecodable bytes are replaced, not raised.
    sample_lines : int, default 200
        Number of physical lines read from the top. When the file is
        longer, the last record of the sample, which may be cut off, is
        not used.

    Returns
    -------
    int
        Number of physical lines before the header, to be passed as
        ``skip_lines`` to ``pl.read_csv`` or ``pl.scan_csv``.

    Raises
    ------
    ValueError
        If the file has no non-empty line, if no line starts a table, or
        if the line just above the detected header is not empty and has
        ``W`` cells, which cannot be told apart from the first line of a
        two-line header.

    Notes
    -----
    Metadata lines with as many cells as the table must be separated from
    it by an empty line. A table whose columns are all text, preceded
    without an empty line by a metadata line with as many cells, is read
    with that metadata line as its header, because it cannot be told
    apart from a CSV without metadata.
    """
    with Path(source).open(encoding=encoding, errors="replace", newline="\n") as file:
        lines = list(islice(file, sample_lines + 1))
    truncated = len(lines) > sample_lines
    lines = lines[:sample_lines]
    while lines and _is_blank(lines[-1]):
        lines.pop()
    if not lines:
        raise ValueError(f"{source}: the file has no non-empty line.")

    for start, line in enumerate(lines):
        if _is_blank(line):
            continue
        records = _parse(lines[start:], separator, quote_char)
        if records is None:
            continue
        if truncated and len(records) > 1:
            records.pop()
        if not _forms_table(records):
            continue
        if start > 0 and not _is_blank(lines[start - 1]):
            previous = _parse(lines[start - 1 : start], separator, quote_char)
            if previous and len(previous[0]) == len(records[0]):
                raise ValueError(
                    f"{source}: line {start} has as many cells as the header "
                    f"candidate on line {start + 1} (skip_lines={start}); it "
                    "may be metadata or the first line of a two-line header. "
                    "Put an empty line after the metadata or pass skip_lines."
                )
        return start

    raise ValueError(
        f"{source}: no line among lines 1-{len(lines)} starts a table with "
        f"two or more columns and at least {_MIN_DATA_ROWS} data rows. "
        "Check separator and quote_char, or pass skip_lines."
    )
