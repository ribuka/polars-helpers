"""Run the Python examples of ``docs/*.md`` and compare their output.

Each Python block of a page runs, from top to bottom, in one namespace with
an empty temporary folder as the working directory. A ``text`` block right
after a Python block is the standard output that block must print.
"""

import contextlib
import io
import re
import tempfile
from pathlib import Path

import pytest

from polars_helpers import _disk

DOCS = Path(__file__).parents[1] / "docs"

_FENCE = re.compile(r"^```(?P<lang>[\w-]*)[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def examples(text: str) -> list[tuple[str, str | None]]:
    """Find the Python blocks of a page and the output they must print.

    Parameters
    ----------
    text : str
        Markdown.

    Returns
    -------
    list[tuple[str, str | None]]
        Code of each Python block, with the body of the ``text`` block that
        follows it with only blank lines between, or ``None``.
    """
    fences = list(_FENCE.finditer(text))
    out: list[tuple[str, str | None]] = []
    for i, fence in enumerate(fences):
        if fence.group("lang") != "python":
            continue
        expected = None
        if i + 1 < len(fences):
            after = fences[i + 1]
            between = text[fence.end() : after.start()]
            if after.group("lang") == "text" and not between.strip():
                expected = after.group("body")
        out.append((fence.group("body"), expected))
    return out


def normalize(output: str) -> str:
    """Drop the trailing spaces of each line and the trailing blank lines.

    Parameters
    ----------
    output : str
        Text.

    Returns
    -------
    str
        Normalized text.
    """
    return "\n".join(line.rstrip() for line in output.splitlines()).rstrip("\n")


def run_page(text: str, name: str) -> None:
    """Run the Python blocks of a page and compare their output.

    Parameters
    ----------
    text : str
        Markdown.
    name : str
        Name of the page, for messages.

    Raises
    ------
    AssertionError
        If a block prints something else than its ``text`` block.
    """
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for n, (code, expected) in enumerate(examples(text), start=1):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, f"{name}[{n}]", "exec"), namespace)  # noqa: S102
        if expected is not None:
            assert normalize(buffer.getvalue()) == normalize(expected), (
                f"block {n} of {name} printed:\n{buffer.getvalue()}"
            )


@pytest.fixture
def docs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in an empty folder, with every default location inside it.

    Parameters
    ----------
    tmp_path : Path
        Temporary folder of the test.
    monkeypatch : pytest.MonkeyPatch
        Patcher.

    Returns
    -------
    Path
        The temporary folder.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return tmp_path


PAGES = sorted(DOCS.glob("*.md"))


def test_there_are_pages() -> None:
    """Find the pages to run."""
    assert PAGES


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_page_runs_and_prints_what_it_says(page: Path, docs_env: Path) -> None:
    """Run every example of a page and compare its output."""
    assert _disk.default_root().is_relative_to(docs_env)
    run_page(page.read_text(encoding="utf-8"), page.name)
    assert _disk.default_root().is_relative_to(docs_env)


def test_wrong_output_fails(docs_env: Path) -> None:
    """Fail when a block prints something else than its text block."""
    good = "```python\nprint('a')  \nprint(1 + 1)\n```\n\n```text\na\n2\n\n```\n"
    run_page(good, "good.md")
    with pytest.raises(AssertionError, match="block 1"):
        run_page(good.replace("a\n2\n", "a\n3\n"), "bad.md")


def test_changed_page_fails(docs_env: Path) -> None:
    """Fail when the output of a real page is changed."""
    page = DOCS / "scan_parquet_relaxed.md"
    text = page.read_text(encoding="utf-8")
    assert "{'x': 0.5, 'flag': True, 'label': None}" in text
    with pytest.raises(AssertionError):
        run_page(text.replace("{'x': 0.5, 'flag': True", "{'x': 9.5, 'flag': True", 1), page.name)


def test_examples_pairs_blocks() -> None:
    """Pair a Python block only with a text block right after it."""
    text = (
        "```python\nprint(1)\n```\n\n```text\n1\n```\n"
        "```python\nx = 1\n```\nprose\n```text\nignored\n```\n"
        "```sh\nls\n```\n"
    )
    assert examples(text) == [("print(1)\n", "1\n"), ("x = 1\n", None)]
