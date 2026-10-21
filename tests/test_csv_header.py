"""Tests for detecting the header line of a CSV preceded by metadata."""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

import polars_helpers as plh
from polars_helpers.utils import detect_csv_skip_lines

BACKGROUND = (
    "Equipment: ABC-123\n"
    "Recipe: TEST_01\n"
    "\n"
    "Time,Pressure,Temperature\n"
    "0,100,25\n"
    "1,101,26\n"
)


def _write(tmp_path: Path, text: str) -> Path:
    """Write CSV text to a file.

    Parameters
    ----------
    tmp_path : Path
        Directory to write into.
    text : str
        File contents.

    Returns
    -------
    Path
        Path of the written file.
    """
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8", newline="")
    return path


@pytest.mark.parametrize(
    ("text", "quote_char", "expected", "columns", "n_rows"),
    [
        pytest.param(
            BACKGROUND, '"', 3, ["Time", "Pressure", "Temperature"], 2, id="background"
        ),
        pytest.param("a,b\n1,2\n3,4\n", '"', 0, ["a", "b"], 2, id="no-metadata"),
        pytest.param(
            "Equipment,ABC-123\nRecipe,TEST_01\n\nTime,Pressure\n0,100\n1,101\n",
            '"',
            3,
            ["Time", "Pressure"],
            2,
            id="key-value-metadata-and-empty-line",
        ),
        pytest.param(
            'Note: "abc\nfoo\n\nTime,P\n0,1\n1,2\n',
            '"',
            3,
            ["Time", "P"],
            2,
            id="unclosed-quote-in-metadata",
        ),
        pytest.param(
            '"meta\n",info\n\nName,Value\n"a",1\nb,2\nc,3\n',
            '"',
            3,
            ["Name", "Value"],
            3,
            id="quoted-metadata-spanning-lines",
        ),
        pytest.param(
            'Name,Value\n"multi\nline",1\na,2\n"x\ny\nz",3\n',
            '"',
            0,
            ["Name", "Value"],
            3,
            id="quoted-data-spanning-lines",
        ),
        pytest.param(
            'Name,Value\n12" pipe,1\nb,2\n',
            None,
            0,
            ["Name", "Value"],
            2,
            id="no-quote-char",
        ),
        pytest.param(
            "Title: list\nOwner: me\n\nName,City\na,x\nb,y\n",
            '"',
            3,
            ["Name", "City"],
            2,
            id="text-table-with-metadata-and-empty-line",
        ),
        pytest.param(
            "Title,list\nName,City\na,x\nb,y\n",
            '"',
            0,
            ["Title", "list"],
            3,
            id="text-table-with-metadata-without-empty-line",
        ),
        pytest.param(
            "Meta: x\r\n\r\nTime,P\r\n0,NA\r\n1,2\r\n",
            '"',
            2,
            ["Time", "P"],
            2,
            id="crlf-and-missing-token",
        ),
    ],
)
def test_detect_csv_skip_lines_finds_header(
    tmp_path: Path,
    text: str,
    quote_char: str | None,
    expected: int,
    columns: list[str],
    n_rows: int,
) -> None:
    """Return the line count that makes ``pl.read_csv`` read the table."""
    path = _write(tmp_path, text)

    skip_lines = detect_csv_skip_lines(path, quote_char=quote_char)

    assert skip_lines == expected
    df = pl.read_csv(path, skip_lines=skip_lines, quote_char=quote_char)
    assert df.columns == columns
    assert df.height == n_rows


@pytest.mark.parametrize(
    ("text", "match"),
    [
        pytest.param("", "no non-empty line", id="empty"),
        pytest.param("\n\n", "no non-empty line", id="only-empty-lines"),
        pytest.param("a\n1\n2\n", "starts a table", id="one-column"),
        pytest.param("Meta: x\n\na,b\n1,2\n", "starts a table", id="too-few-rows"),
        pytest.param("a,b\n1,2\n\n3,4\n", "starts a table", id="empty-line-in-table"),
        pytest.param(
            "Equipment,ABC-123\nRecipe,TEST_01\nTime,Pressure\n0,100\n1,101\n",
            "line 2",
            id="same-width-metadata-without-empty-line",
        ),
        pytest.param(
            "Time,Pressure\ns,kPa\n0,100\n1,101\n",
            "two-line header",
            id="two-line-header",
        ),
    ],
)
def test_detect_csv_skip_lines_raises_when_unsure(
    tmp_path: Path, text: str, match: str
) -> None:
    """Raise instead of guessing when no line is clearly the header."""
    path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=match):
        detect_csv_skip_lines(path)


def test_detect_csv_skip_lines_ignores_record_cut_by_sample(tmp_path: Path) -> None:
    """Drop the last record of a sample that ends inside a quoted cell."""
    path = _write(tmp_path, 'Meta: x\n\nName,Value\na,1\nb,2\n"c\nd",3\n')

    assert detect_csv_skip_lines(path, sample_lines=6) == 2


def test_detect_csv_skip_lines_uses_separator(tmp_path: Path) -> None:
    """Split cells with the given separator."""
    path = _write(tmp_path, "Meta: a,b\n\nTime;P\n0;1\n1;2\n")

    assert detect_csv_skip_lines(path, separator=";") == 2


def test_read_csv_detects_with_separator_and_quote_char(tmp_path: Path) -> None:
    """Use ``separator`` and ``quote_char`` for both detection and reading."""
    path = _write(tmp_path, "Meta: x\n\nName;Value\n'a;b';1\nc;2\n")

    df = plh.read_csv(path, separator=";", quote_char="'")

    assert df.columns == ["Name", "Value"]
    assert df["Name"].to_list() == ["a;b", "c"]


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"skip_lines": 1}, id="skip_lines"),
        pytest.param({"skip_rows": 1}, id="skip_rows"),
    ],
)
def test_read_csv_passes_explicit_skip_through(
    tmp_path: Path, kwargs: dict[str, int]
) -> None:
    """Skip detection when the caller says how many lines to skip."""
    path = _write(tmp_path, "a,b\nc,d\n1,2\n")

    df = plh.read_csv(path, **kwargs)

    assert df.columns == ["c", "d"]


def test_read_csv_without_header_skips_detection(tmp_path: Path) -> None:
    """Read every line as data when ``has_header`` is False."""
    path = _write(tmp_path, "Meta: x\n")

    df = plh.read_csv(path, has_header=False)

    assert df.height == 1


def test_scan_csv_returns_lazy_frame(tmp_path: Path) -> None:
    """Return a ``LazyFrame`` that reads the table below the metadata."""
    path = _write(tmp_path, BACKGROUND)

    result = plh.scan_csv(path)

    assert isinstance(result, pl.LazyFrame)
    assert result.collect().equals(
        pl.DataFrame({"Time": [0, 1], "Pressure": [100, 101], "Temperature": [25, 26]})
    )


def test_detect_csv_skip_lines_skips_unparsable_candidate(tmp_path: Path) -> None:
    """Try later candidates when the lines from an earlier one fail to parse."""
    path = tmp_path / "data.csv"
    path.write_bytes(b"Meta: x\ry\n\na,b\n1,2\n3,4\n")

    assert detect_csv_skip_lines(path) == 2


def test_detect_csv_skip_lines_accepts_long_cells(tmp_path: Path) -> None:
    """Accept cells longer than the default ``csv.field_size_limit``."""
    limit = csv.field_size_limit()
    long_cell = "x" * (limit + 1)
    path = _write(tmp_path, f"Meta: x\n\nname,value\n{long_cell},1\ny,2\n")

    assert detect_csv_skip_lines(path) == 2
    assert csv.field_size_limit() == limit


def test_read_csv_detects_with_encoding(tmp_path: Path) -> None:
    """Decode the file with the given ``encoding`` for the detection too."""
    path = tmp_path / "data.csv"
    path.write_text("Metadata\n\na,b\n1,2\n3,4\n", encoding="utf-16")

    df = plh.read_csv(path, encoding="utf-16")

    assert df.columns == ["a", "b"]
    assert df.height == 2


def test_detect_csv_skip_lines_accepts_long_cells_concurrently(tmp_path: Path) -> None:
    """Keep long cells accepted and the limit restored across threads."""
    limit = csv.field_size_limit()
    path = _write(tmp_path, f"name,value\n{'x' * (limit + 1)},1\ny,2\n")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: detect_csv_skip_lines(path), range(32)))

    assert results == [0] * 32
    assert csv.field_size_limit() == limit
