"""Header-row detection for CSV files that start with metadata lines."""

from __future__ import annotations

import csv
import re
from datetime import datetime, time
from enum import Enum
from itertools import islice
from pathlib import Path

# Fewest data lines that must follow the detected header.
_MIN_DATA_LINES = 2
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
# Cells that mean "missing" and therefore carry no type information.
_NULL_TOKENS = frozenset({"", "-", "na", "n/a", "nan", "null", "none"})


class _Kind(Enum):
    """Value kind used to tell a header line from a data line."""

    NUMBER = "number"
    TEMPORAL = "temporal"
    TEXT = "text"


def _is_temporal(value: str) -> bool:
    """Tell whether a cell parses as an ISO-like date, datetime, or time.

    Parameters
    ----------
    value : str
        Stripped, non-null cell text.

    Returns
    -------
    bool
        ``True`` if the cell is a date, datetime, or time, with ``-`` or
        ``/`` as the date separator.
    """
    candidate = value.replace("/", "-")
    for parse in (datetime.fromisoformat, time.fromisoformat):
        try:
            parse(candidate)
        except ValueError:
            continue
        return True
    return False


def _kind(value: str) -> _Kind | None:
    """Classify one cell.

    Parameters
    ----------
    value : str
        Raw cell text.

    Returns
    -------
    _Kind | None
        The kind of the cell, or ``None`` if it is empty or a null token.
    """
    stripped = value.strip()
    if stripped.lower() in _NULL_TOKENS:
        return None
    if _NUMBER.fullmatch(stripped):
        return _Kind.NUMBER
    if _is_temporal(stripped):
        return _Kind.TEMPORAL
    return _Kind.TEXT


def _column_types(kinds: list[set[_Kind]]) -> list[_Kind | None]:
    """Reduce the kinds seen in each column to a single column type.

    Parameters
    ----------
    kinds : list[set[_Kind]]
        Kinds of the non-null cells seen in each column.

    Returns
    -------
    list[_Kind | None]
        ``NUMBER`` or ``TEMPORAL`` for a column whose cells all have that
        kind, otherwise ``None`` (the column gives no type cue).
    """
    types: list[_Kind | None] = []
    for column in kinds:
        only = next(iter(column)) if len(column) == 1 else None
        types.append(only if only is not _Kind.TEXT else None)
    return types


def _is_header_like(fields: list[str], types: list[_Kind | None]) -> bool:
    """Tell whether a line contradicts the column types of the data below.

    Parameters
    ----------
    fields : list[str]
        Cells of the line.
    types : list[_Kind | None]
        Column types inferred from the lines below it.

    Returns
    -------
    bool
        ``True`` if a typed column holds a non-null cell of another kind.
    """
    for value, column_type in zip(fields, types, strict=True):
        if column_type is None:
            continue
        kind = _kind(value)
        if kind is not None and kind is not column_type:
            return True
    return False


def _read_sample(
    source: str | Path, separator: str, encoding: str, sample_lines: int
) -> list[list[str]]:
    """Parse the first physical lines of a file one line at a time.

    Each line is parsed on its own so that an unclosed ``"`` in the metadata
    never swallows the lines after it.

    Parameters
    ----------
    source : str | Path
        CSV file path.
    separator : str
        Field separator.
    encoding : str
        Text encoding of the file.
    sample_lines : int
        Number of physical lines to read.

    Returns
    -------
    list[list[str]]
        Cells of each line, with trailing empty lines removed. An empty line
        has no cells.
    """
    with Path(source).open(encoding=encoding, newline="\n") as file:
        lines = [line.rstrip("\r\n") for line in islice(file, sample_lines)]
    rows = [next(csv.reader([line], delimiter=separator), []) for line in lines]
    while rows and not rows[-1]:
        rows.pop()
    return rows


def _describe(rows: list[list[str]], start: int, stop: int) -> str:
    """Format sample lines for an error message.

    Parameters
    ----------
    rows : list[list[str]]
        Parsed sample lines.
    start : int
        First line number to show (0-based, clipped to the sample).
    stop : int
        Line number after the last one to show.

    Returns
    -------
    str
        One ``line <n> (<count> fields): <cells>`` entry per line.
    """
    return "\n".join(
        f"  line {number} ({len(rows[number])} fields): {rows[number]}"
        for number in range(max(start, 0), min(stop, len(rows)))
    )


def detect_csv_skip_lines(
    source: str | Path,
    *,
    separator: str = ",",
    encoding: str = "utf-8-sig",
    sample_lines: int = 200,
) -> int:
    """Find the header line of a CSV file that may start with metadata lines.

    Only the first ``sample_lines`` physical lines are read. The field count
    of the last line is taken as the table width, the lines of that width
    directly above them form the candidate block, and the header is the
    lowest line in the block whose cells contradict the column types (number
    or date/time) of the lines below it. If no line does (for example every
    column is text, or the column names are numbers), the top of the block is
    the header; the block always starts at the file start or after an empty
    line or a change of field count.

    Pass the result as ``skip_lines=`` (not ``skip_rows=``) to
    ``pl.read_csv`` or ``pl.scan_csv``: it counts physical lines, and
    ``skip_rows`` would count a quoted line break in the metadata differently.

    Known limitations: metadata with the same field count as the table and no
    empty line before the header is told apart only through type cues, so
    with all-text columns its first line is taken as the header; and a
    two-line header directly under three or more such metadata lines is read
    as metadata plus a one-line header.

    Parameters
    ----------
    source : str | Path
        CSV file path.
    separator : str, default ","
        Field separator.
    encoding : str, default "utf-8-sig"
        Text encoding of the file; the default also strips a UTF-8 BOM.
    sample_lines : int, default 200
        Number of physical lines to inspect.

    Returns
    -------
    int
        Number of physical lines before the header line.

    Raises
    ------
    ValueError
        If the file is empty; the last line has fewer than 2 fields; fewer
        than 2 data lines follow the header (also when the last lines do not
        share one field count); the header
        looks like two lines (for example names and units); or lines that fit
        the data types sit above the detected header. The message shows the
        width and the candidate lines so that ``skip_lines`` can be given by
        hand instead.
    """
    rows = _read_sample(source, separator, encoding, sample_lines)
    if not rows:
        raise ValueError(f"{source} has no lines to detect a CSV header from.")

    end = len(rows)
    width = len(rows[-1])
    if width < 2:
        raise ValueError(
            f"{source} ends with a line of {width} field(s); "
            "a table needs 2 or more.\n" + _describe(rows, end - 3, end)
        )

    start = end
    while start > 0 and len(rows[start - 1]) == width:
        start -= 1
    if end - start < 1 + _MIN_DATA_LINES:
        raise ValueError(
            f"Too few lines of the same field count end {source} "
            f"(width {width}):\n{_describe(rows, start - 2, end)}"
        )

    kinds: list[set[_Kind]] = [set() for _ in range(width)]
    header = start
    for number in range(end - 1, start - 1, -1):
        if _is_header_like(rows[number], _column_types(kinds)):
            header = number
            break
        for column, value in zip(kinds, rows[number], strict=True):
            kind = _kind(value)
            if kind is not None:
                column.add(kind)

    candidates = _describe(rows, header - 2, header + 3)
    if end - header - 1 < _MIN_DATA_LINES:
        raise ValueError(
            f"Too few data lines follow the header candidate of {source} "
            f"(width {width}):\n{candidates}"
        )

    if header > start:
        types = _column_types(kinds)
        above = [_is_header_like(rows[n], types) for n in range(start, header)]
        if not all(above):
            raise ValueError(
                f"Lines that fit the data sit above the header candidate of "
                f"{source} (width {width}):\n{_describe(rows, start, header + 3)}"
            )
        if len(above) == 1:
            raise ValueError(
                f"{source} looks like it has a two-line header "
                f"(width {width}):\n{candidates}"
            )
    return header
