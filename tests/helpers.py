"""Child processes for the tests of the disk helpers.

A child runs ``code`` after a prelude defining ``wait_for_parent()``. The
child calls it once set up; it waits for a line from the parent and either
returns, so that the child exits normally, or kills the child at once. The
child kills itself because on Windows ``sys.executable`` of a virtual
environment may be a launcher, and killing the launcher would leave the real
interpreter running.
"""

import subprocess
import sys
from pathlib import Path

_PRELUDE = """
import os
import signal
import sys


def wait_for_parent():
    if sys.stdin.readline().strip() == "kill":
        os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))

"""


def start(code: str, *args: str, cwd: Path | None = None) -> subprocess.Popen[str]:
    """Start a child process.

    Parameters
    ----------
    code : str
        Code to run after the prelude; it receives ``args`` from
        ``sys.argv[1]``.
    *args : str
        Arguments of the child.
    cwd : Path | None, default None
        Working directory of the child.

    Returns
    -------
    subprocess.Popen[str]
        The child.
    """
    return subprocess.Popen(
        [sys.executable, "-c", _PRELUDE + code, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )


def ready(proc: subprocess.Popen[str]) -> str:
    """Wait for the first line a child prints.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Child process.

    Returns
    -------
    str
        The line, stripped.

    Raises
    ------
    AssertionError
        If the child ends without printing a line.
    """
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line:
        _, err = proc.communicate(timeout=60)
        raise AssertionError(f"the child ended before it was ready:\n{err}")
    return line


def _send(proc: subprocess.Popen[str], line: str) -> tuple[str, str]:
    """Send a line to a child and wait for it to end.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Child process.
    line : str
        Line to send.

    Returns
    -------
    tuple[str, str]
        The rest of its standard output and its standard error.
    """
    return proc.communicate(line + "\n", timeout=120)


def kill(proc: subprocess.Popen[str]) -> None:
    """Make a child kill itself, without any cleanup, and wait for it.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Child process waiting in ``wait_for_parent()``.
    """
    _send(proc, "kill")
    assert proc.returncode != 0


def finish(proc: subprocess.Popen[str]) -> str:
    """Let a child exit normally and wait for it.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Child process waiting in ``wait_for_parent()``.

    Returns
    -------
    str
        The rest of its standard output.

    Raises
    ------
    AssertionError
        If the child fails.
    """
    out, err = _send(proc, "")
    if proc.returncode != 0:
        raise AssertionError(f"the child failed:\n{err}")
    return out


def run(code: str, *args: str, cwd: Path | None = None) -> str:
    """Run a child process to its normal end.

    Parameters
    ----------
    code : str
        Code to run after the prelude; ``wait_for_parent()`` need not be
        called.
    *args : str
        Arguments of the child.
    cwd : Path | None, default None
        Working directory of the child.

    Returns
    -------
    str
        Its standard output.

    Raises
    ------
    AssertionError
        If the child fails.
    """
    proc = start(code, *args, cwd=cwd)
    out, err = proc.communicate("\n", timeout=120)
    if proc.returncode != 0:
        raise AssertionError(f"the child failed:\n{err}")
    return out
