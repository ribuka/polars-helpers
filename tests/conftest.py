"""Shared fixtures of the tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the default root of the disk helpers into a temporary folder.

    Every test, and every child process it starts, sees ``LOCALAPPDATA``,
    ``XDG_CACHE_HOME``, ``HOME``, and ``USERPROFILE`` in a folder of its own,
    so that no test reads or writes the real user cache.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory of temporary folders.
    monkeypatch : pytest.MonkeyPatch
        Patcher of the environment.

    Yields
    ------
    Path
        Folder holding the redirected locations.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    yield home
