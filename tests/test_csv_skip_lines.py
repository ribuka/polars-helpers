"""Tests for CSV header detection and the CSV readers built on it."""

from pathlib import Path

import polars as pl
import pytest

import polars_helpers as plh
from polars_helpers.utils import detect_csv_skip_lines

_TABLE = "Time,Pressure,Temperature\n0,100,25\n1,101,26\n2,102,27\n"
_COLUMNS = ["Time", "Pressure", "Temperature"]


def _write(tmp_path: Path, text: str, encoding: str = "utf-8") -> Path:
    """Write CSV text to a temporary file.

    Parameters
    ----------
    tmp_path : Path
        Directory to write into.
    text : str
        File contents.
    encoding : str, default "utf-8"
        Encoding to write with.

    Returns
    -------
    Path
        Path of the written file.
    """
    path = tmp_path / "input.csv"
    path.write_text(text, encoding=encoding, newline="")
    return path


@pytest.mark.parametrize(
    ("text", "separator", "encoding", "expected_skip", "expected_columns"),
    [
        pytest.param(
            "Equipment: ABC-123\nRecipe: TEST_01\nDate: 2026-10-21\n\n" + _TABLE,
            ",",
            "utf-8",
            4,
            _COLUMNS,
            id="basic",
        ),
        pytest.param(_TABLE, ",", "utf-8", 0, _COLUMNS, id="no-metadata"),
        pytest.param(
            "Equipment,ABC-123\nDate,2026-10-21\n\n" + _TABLE,
            ",",
            "utf-8",
            3,
            _COLUMNS,
            id="key-value-metadata-and-empty-line",
        ),
        pytest.param(
            "Equipment,ABC-123\nRecipe,TEST_01\nTime,Pressure\n0,100\n1,101\n",
            ",",
            "utf-8",
            2,
            ["Time", "Pressure"],
            id="same-width-metadata-without-empty-line",
        ),
        pytest.param(
            "Equipment: ABC-123\n\nTime,Pressure,\n0,100,\n1,101,\n",
            ",",
            "utf-8",
            2,
            ["Time", "Pressure", ""],
            id="trailing-separator",
        ),
        pytest.param(
            "Equipment\tABC-123\n\n" + _TABLE.replace(",", "\t"),
            "\t",
            "utf-8",
            2,
            _COLUMNS,
            id="tab-separated",
        ),
        pytest.param(
            "Equipment: ABC-123\r\n\r\n" + _TABLE.replace("\n", "\r\n"),
            ",",
            "utf-8-sig",
            2,
            _COLUMNS,
            id="bom-and-crlf",
        ),
        pytest.param(
            'Note: "line1\nline2"\n\n' + _TABLE,
            ",",
            "utf-8",
            3,
            _COLUMNS,
            id="unclosed-quote-in-metadata",
        ),
        pytest.param(
            "".join(f"Key{i}: value{i}\n" for i in range(150)) + _TABLE,
            ",",
            "utf-8",
            150,
            _COLUMNS,
            id="150-metadata-lines",
        ),
        pytest.param(
            "Name,City\nalice,tokyo\nbob,osaka\n",
            ",",
            "utf-8",
            0,
            ["Name", "City"],
            id="all-text-columns",
        ),
        pytest.param(
            "Equipment: ABC-123\n\nTime,Pressure\n0,NA\n1,\n2,102\n",
            ",",
            "utf-8",
            2,
            ["Time", "Pressure"],
            id="null-tokens-in-data",
        ),
        pytest.param(
            "Date: 2026-10-21\n\nTimestamp,Value\n"
            "2026-10-21 10:00:00,1\n2026-10-21 10:00:01,2\n",
            ",",
            "utf-8",
            2,
            ["Timestamp", "Value"],
            id="datetime-column",
        ),
    ],
)
def test_detect_csv_skip_lines_finds_header(
    tmp_path: Path,
    text: str,
    separator: str,
    encoding: str,
    expected_skip: int,
    expected_columns: list[str],
) -> None:
    """Return the header line, which polars then reads as the column names."""
    path = _write(tmp_path, text, encoding)

    skip = detect_csv_skip_lines(path, separator=separator, encoding=encoding)

    assert skip == expected_skip
    frame = pl.read_csv(path, skip_lines=skip, separator=separator)
    assert frame.columns == expected_columns


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("", "no lines", id="empty"),
        pytest.param("\n\n", "no lines", id="only-empty-lines"),
        pytest.param("Time\n0\n1\n2\n", "2 or more", id="one-column"),
        pytest.param(
            "Time,Pressure\n0,100\n1,101,9\n", "Too few", id="ragged-tail"
        ),
        pytest.param(
            "Time,Pressure\n0,100\n1,101\nA,B\n2,102\n",
            "Too few",
            id="header-candidate-near-the-end",
        ),
        pytest.param(
            "Equipment: ABC-123\n\nTime,Pressure\ns,kPa\n0,100\n1,101\n",
            "two-line header",
            id="two-line-header",
        ),
        pytest.param(
            "Equipment: ABC-123\n\nTime,Pressure\n0,100\n",
            "Too few",
            id="too-few-data-lines",
        ),
        pytest.param(
            "Time,Pressure\n0,100\n1,101\n2,ERR\n3,103\n4,104\n",
            "fit the data",
            id="text-inside-numeric-data",
        ),
        pytest.param(
            "Name,Value\na,1\n\nb,2\nc,3\nd,4\ne,5\n",
            "fit the data",
            id="empty-line-inside-data",
        ),
        pytest.param(
            "Time,Pressure\n0,100\n1,101,9\n2,102\n3,103\n4,104\n",
            "fit the data",
            id="ragged-line-inside-data",
        ),
        pytest.param(
            'Time,Note\n0,"a\nb"\n1,c\n2,d\n3,e\n',
            "fit the data",
            id="quoted-line-break-inside-data",
        ),
        pytest.param(
            'Name,Value\n"multi\nline",1\na,2\nb,3\nc,4\n',
            "spans lines",
            id="quoted-line-break-in-first-column",
        ),
        pytest.param(
            "Key,Value\nk1,v1\n\nName,City\nalice,tokyo\nbob,osaka\n",
            "fit the data",
            id="all-text-same-width-metadata-before-empty-line",
        ),
    ],
)
def test_detect_csv_skip_lines_raises_when_unsure(
    tmp_path: Path, text: str, message: str
) -> None:
    """Raise instead of guessing when the header line is not clear."""
    path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=message):
        detect_csv_skip_lines(path)


def test_detect_csv_skip_lines_reads_only_sample_lines(tmp_path: Path) -> None:
    """Ignore the lines after ``sample_lines``."""
    path = _write(tmp_path, "Equipment: ABC-123\n\n" + _TABLE + "3,103\n" * 10)

    assert detect_csv_skip_lines(path, sample_lines=5) == 2


def test_detect_csv_skip_lines_accepts_other_encodings(tmp_path: Path) -> None:
    """Decode the sample with the given encoding."""
    path = _write(tmp_path, "装置: ABC-123\n\n" + _TABLE, encoding="cp932")

    assert detect_csv_skip_lines(path, encoding="cp932") == 2


def test_read_csv_skips_metadata(tmp_path: Path) -> None:
    """Read the table under the metadata and pass other arguments through."""
    path = _write(tmp_path, "Equipment: ABC-123\n\n" + _TABLE)

    frame = plh.read_csv(path, columns=["Time", "Pressure"])

    assert frame.equals(
        pl.DataFrame({"Time": [0, 1, 2], "Pressure": [100, 101, 102]})
    )


def test_scan_csv_returns_lazy_frame(tmp_path: Path) -> None:
    """Scan the table under the metadata lazily."""
    path = _write(tmp_path, "Equipment\tABC-123\n\n" + _TABLE.replace(",", "\t"))

    result = plh.scan_csv(path, separator="\t")

    assert isinstance(result, pl.LazyFrame)
    assert result.collect().equals(pl.read_csv(_write(tmp_path, _TABLE)))


@pytest.mark.parametrize("reader", [plh.read_csv, plh.scan_csv])
@pytest.mark.parametrize(
    "kwargs",
    [{"skip_lines": 1}, {"skip_rows": 1}, {"has_header": False}],
    ids=["skip_lines", "skip_rows", "has_header"],
)
def test_csv_readers_reject_manual_header_arguments(
    tmp_path: Path, reader: object, kwargs: dict[str, object]
) -> None:
    """Reject arguments that contradict the header detection."""
    path = _write(tmp_path, _TABLE)

    with pytest.raises(TypeError):
        reader(path, **kwargs)  # type: ignore[operator]
