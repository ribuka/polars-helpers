"""Shared disk handling of the helpers that write files.

This module decides where the helpers write by default, checks and creates
the directory a caller gives, and keeps one session per process and root:
the files a call registers are deleted when the process exits normally, and
what a killed process left is collected by a later call on the same root.

A session lives directly in its root as two files:

- ``.polars-helpers-session-<id>.lock``, locked by the process while it
  lives; the OS releases the lock when the process ends, even if killed.
- ``.polars-helpers-session-<id>.json``, the marker, the user, and the
  files and directories registered to delete on exit.

Nothing here is specific to one helper.
"""

from __future__ import annotations

import atexit
import contextlib
import getpass
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

# Name of the directory under the user cache directory of the OS.
_APP = "polars-helpers"

# Prefix of the files of a session, directly in its root.
SESSION_PREFIX = ".polars-helpers-session-"

# Marker in the JSON file of a session, so that only files this package
# wrote are read and collected.
_SESSION_MARKER = "polars-helpers-session"

# Interval of the retries of a blocking lock on Windows, in seconds.
_LOCK_RETRY_SECONDS = 0.02

# Longest time, in seconds, a lock file may refuse to open on Windows while
# another process deletes it, before the refusal is raised.
_PENDING_DELETE_SECONDS = 2.0


def default_root() -> Path:
    """Find the default directory of the files the helpers write.

    Returns
    -------
    Path
        The user cache directory of the OS: ``%LOCALAPPDATA%`` (or
        ``~/AppData/Local``) ``/polars-helpers/cache`` on Windows,
        ``~/Library/Caches/polars-helpers`` on macOS, and
        ``$XDG_CACHE_HOME`` (or ``~/.cache``) ``/polars-helpers`` elsewhere.
        The temporary directory of the OS is never used, since it may be
        cleared at logoff, by the OS, or be shared by every user.
    """
    return _default_root_for(sys.platform, os.environ, Path.home)


def _default_root_for(
    platform: str, environ: Mapping[str, str], home: Callable[[], Path]
) -> Path:
    """Find the default root for a platform and environment.

    Parameters
    ----------
    platform : str
        Value of ``sys.platform``.
    environ : Mapping[str, str]
        Environment variables.
    home : Callable[[], Path]
        Function returning the home directory of the user, called only when
        needed.

    Returns
    -------
    Path
        Default root, as ``default_root`` describes. An environment variable
        that is unset, empty, or a relative path is ignored.
    """

    def absolute_env(name: str) -> Path | None:
        value = environ.get(name, "")
        if value and Path(value).is_absolute():
            return Path(value)
        return None

    if platform == "win32":
        base = absolute_env("LOCALAPPDATA") or home() / "AppData" / "Local"
        return base / _APP / "cache"
    if platform == "darwin":
        return home() / "Library" / "Caches" / _APP
    base = absolute_env("XDG_CACHE_HOME") or home() / ".cache"
    return base / _APP


def resolve_root(directory: str | Path | None) -> Path:
    """Check the directory a caller gave and make it absolute.

    Nothing is created or written.

    Parameters
    ----------
    directory : str | Path | None
        Directory given by the caller, or ``None`` for ``default_root()``.
        A relative path is taken from the current working directory with
        ``Path.absolute()``; ``~`` and environment variables are not
        expanded.

    Returns
    -------
    Path
        Absolute path of the root.

    Raises
    ------
    TypeError
        If ``directory`` is not a ``str``, ``Path``, or ``None``.
    ValueError
        If ``directory`` is an empty string.
    NotADirectoryError
        If ``directory`` is an existing file.
    """
    if directory is None:
        return default_root()
    if not isinstance(directory, (str, Path)):
        raise TypeError(f"directory must be a str, Path, or None, got {type(directory)}")
    if isinstance(directory, str) and not directory:
        raise ValueError("directory must not be an empty string")
    path = Path(directory).absolute()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"directory is not a directory: {path}")
    return path


def make_directory(path: Path) -> list[Path]:
    """Create a directory with its parents.

    Parameters
    ----------
    path : Path
        Absolute path of the directory.

    Returns
    -------
    list[Path]
        The directories this call created, the innermost first, so that
        they can be deleted in that order once empty.

    Raises
    ------
    OSError
        If the directory cannot be created, for example without permission.
        No other directory is tried.
    """
    created: list[Path] = []
    ancestor = path
    while not ancestor.exists():
        created.append(ancestor)
        ancestor = ancestor.parent
    path.mkdir(parents=True, exist_ok=True)
    return created


def delete_paths(files: Iterable[Path], directories: Iterable[Path] = ()) -> None:
    """Delete files, then the directories that are left empty.

    Files and directories that cannot be deleted, for example because they
    are open or not empty, are skipped without raising.

    Parameters
    ----------
    files : Iterable[Path]
        Files to delete; missing ones are ignored.
    directories : Iterable[Path], default ()
        Directories to delete if empty, the innermost first.
    """
    for file in files:
        try:
            file.unlink(missing_ok=True)
        except OSError:
            pass
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def user_id() -> str:
    r"""Identify the user running the process.

    Returns
    -------
    str
        ``uid-<uid>`` on POSIX, ``<domain>\<user>`` on Windows.
    """
    if sys.platform == "win32":
        return f"{os.environ.get('USERDOMAIN', '')}\\{getpass.getuser()}"
    return f"uid-{os.getuid()}"


def user_tag() -> str:
    """Give a short name of the user to put in file names.

    Returns
    -------
    str
        First 8 hexadecimal digits of the SHA-256 of ``user_id()``.
    """
    return hashlib.sha256(user_id().encode()).hexdigest()[:8]


def _try_lock(fd: int) -> bool:
    """Try to lock an open file without waiting.

    Parameters
    ----------
    fd : int
        Descriptor of the file.

    Returns
    -------
    bool
        Whether the lock was taken.
    """
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _wait_lock(fd: int) -> None:
    """Lock an open file, waiting until it is free.

    Parameters
    ----------
    fd : int
        Descriptor of the file.
    """
    if sys.platform == "win32":
        while not _try_lock(fd):
            time.sleep(_LOCK_RETRY_SECONDS)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX)


def unlock_file(fd: int) -> None:
    """Release a lock taken by ``lock_file`` and close its file.

    Parameters
    ----------
    fd : int
        Descriptor returned by ``lock_file``.
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        os.close(fd)


def lock_file(path: Path, *, blocking: bool, create: bool = True) -> int | None:
    """Take an exclusive lock on a file, between processes and threads.

    The OS releases the lock when the process ends, even if it is killed.
    The file may be deleted by another process once it holds and releases
    the lock; the lock is taken again until it holds on the file that is
    at ``path``.

    Parameters
    ----------
    path : Path
        Lock file.
    blocking : bool
        Whether to wait until the lock is free.
    create : bool, default True
        Whether to create the file if it is missing.

    Returns
    -------
    int | None
        Descriptor of the locked file, to pass to ``unlock_file``; ``None``
        if ``blocking`` is false and the lock is held by someone else or the
        file cannot be opened, or if the file is missing and ``create`` is
        false.

    Raises
    ------
    PermissionError
        If ``blocking`` is true and the file cannot be opened; on Windows,
        after retrying for a moment in case the file is being deleted.
    """
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    if create:
        flags |= os.O_CREAT
    denied_since: float | None = None
    while True:
        try:
            fd = os.open(path, flags, 0o600)
        except FileNotFoundError:
            if create:
                raise
            return None
        except PermissionError:
            # A file of another user, or one being deleted, is skipped
            # without waiting.
            if not blocking:
                return None
            # On Windows, a file that is being deleted cannot be opened until
            # it is gone, which takes a moment; a denial that lasts is a real
            # lack of permission and is raised.
            if sys.platform != "win32":
                raise
            now = time.monotonic()
            if denied_since is None:
                denied_since = now
            if now - denied_since >= _PENDING_DELETE_SECONDS:
                raise
            time.sleep(_LOCK_RETRY_SECONDS)
            continue
        if blocking:
            _wait_lock(fd)
        elif not _try_lock(fd):
            os.close(fd)
            return None
        try:
            current = os.stat(path)
        except FileNotFoundError:
            current = None
        if current is not None and os.path.samestat(os.fstat(fd), current):
            return fd
        unlock_file(fd)
        if not create:
            return None


def unlock_and_delete(fd: int, path: Path) -> None:
    """Delete a lock file while it cannot be taken, then release its lock.

    The file is deleted only by the holder of its lock, so a process that
    holds the lock on the file at ``path``, as ``lock_file`` checks, keeps
    it until it releases it. On POSIX the file is deleted before the lock is
    released. On Windows an open file cannot be deleted, so the lock is
    released first and the deletion fails, leaving the file, if another
    process opened it in the meantime.

    Parameters
    ----------
    fd : int
        Descriptor returned by ``lock_file`` for ``path``.
    path : Path
        The lock file.
    """
    if sys.platform == "win32":
        unlock_file(fd)
        delete_paths([path])
        return
    delete_paths([path])
    unlock_file(fd)


def write_json(path: Path, data: object) -> None:
    """Write a JSON file atomically.

    Parameters
    ----------
    path : Path
        File to write; it is replaced in one step, so a reader never sees
        it half written.
    data : object
        Value to write.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        delete_paths([tmp])
        raise


def read_json(path: Path) -> dict[str, object] | None:
    """Read a JSON object written by ``write_json``.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    dict[str, object] | None
        The object, or ``None`` if the file is missing or not a JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Session:
    """Files of one process in one root, deleted when the process exits.

    Use ``get_session`` to get the session of the current process for a
    root instead of creating one.

    Parameters
    ----------
    root : Path
        Absolute path of an existing root.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.id = uuid.uuid4().hex[:16]
        self.pid = os.getpid()
        self.lock_path = root / f"{SESSION_PREFIX}{self.id}.lock"
        self.json_path = root / f"{SESSION_PREFIX}{self.id}.json"
        self._files: list[str] = []
        self._known: set[str] = set()
        self._directories: list[str] = []
        self._callbacks: list[Callable[[], None]] = []
        self._mutex = threading.Lock()
        self._closed = False
        fd = lock_file(self.lock_path, blocking=True)
        assert fd is not None
        self._fd = fd
        try:
            self._save()
        except BaseException:
            unlock_and_delete(self._fd, self.lock_path)
            raise

    def _save(self) -> None:
        """Write the JSON file of the session."""
        write_json(
            self.json_path,
            {
                "marker": _SESSION_MARKER,
                "user": user_id(),
                "pid": self.pid,
                "files": self._files,
                "directories": self._directories,
            },
        )

    def register(self, files: Iterable[Path]) -> None:
        """Register files to delete when the process exits.

        Register a file before writing it, so that it is collected even if
        the process is killed while writing it.

        Parameters
        ----------
        files : Iterable[Path]
            Files, in the root or under it.
        """
        self._register(files, ())

    def register_directories(self, directories: Iterable[Path]) -> None:
        """Register directories to delete on exit once they are empty.

        Parameters
        ----------
        directories : Iterable[Path]
            Directories, the innermost first, as ``make_directory`` returns
            them. They are deleted after the files and the session files.
        """
        self._register((), directories)

    def _register(self, files: Iterable[Path], directories: Iterable[Path]) -> None:
        """Register files and directories in one write of the JSON file.

        If the JSON file cannot be written, nothing of this call stays
        registered and the error is raised.

        Parameters
        ----------
        files : Iterable[Path]
            Files, as for ``register``.
        directories : Iterable[Path]
            Directories, as for ``register_directories``.
        """
        with self._mutex:
            new_files = list(dict.fromkeys(str(f) for f in files if str(f) not in self._known))
            new_directories = list(
                dict.fromkeys(str(d) for d in directories if str(d) not in self._directories)
            )
            if not new_files and not new_directories:
                return
            self._files.extend(new_files)
            self._known.update(new_files)
            self._directories.extend(new_directories)
            try:
                self._save()
            except BaseException:
                del self._files[len(self._files) - len(new_files) :]
                self._known.difference_update(new_files)
                del self._directories[len(self._directories) - len(new_directories) :]
                raise

    def discard(self) -> None:
        """Delete the session files of a session that nothing uses.

        For a session that is not yet given out by ``get_session``, such as
        one ``register_on_exit`` created and failed to register in.
        """
        with self._mutex:
            self._closed = True
        delete_paths([self.json_path])
        unlock_and_delete(self._fd, self.lock_path)

    def at_exit(self, callback: Callable[[], None]) -> None:
        """Run a function on exit before the registered files are deleted.

        Parameters
        ----------
        callback : Callable[[], None]
            Function to run; an exception it raises is ignored.
        """
        with self._mutex:
            self._callbacks.append(callback)

    def close(self) -> None:
        """Delete the registered files, the session files, and directories.

        Called on normal exit by ``atexit``. If a registered file cannot be
        deleted, for example because it is open, the session files are kept
        with only the files left, so that a later call on the same root
        collects them once the lock is released; the directories are kept.
        """
        with self._mutex:
            if self._closed or os.getpid() != self.pid:
                return
            self._closed = True
            callbacks = list(self._callbacks)
        for callback in callbacks:
            with contextlib.suppress(Exception):
                callback()
        left = _delete_registered(self._files)
        if left:
            self._files = left
            with contextlib.suppress(OSError):
                self._save()
            unlock_file(self._fd)
            return
        delete_paths([self.json_path])
        unlock_and_delete(self._fd, self.lock_path)
        delete_paths([], (Path(d) for d in self._directories))


def _delete_registered(files: list[str]) -> list[str]:
    """Delete registered files and tell which are left.

    Parameters
    ----------
    files : list[str]
        Paths of the files.

    Returns
    -------
    list[str]
        Paths of the files that may still exist: a file is dropped only once
        it is deleted or known to be missing, never because checking it
        failed, for example without permission.
    """
    left: list[str] = []
    for file in files:
        try:
            Path(file).unlink()
        except (FileNotFoundError, NotADirectoryError):
            pass
        except OSError:
            left.append(file)
    return left


_sessions: dict[tuple[int, str], Session] = {}
_sessions_mutex = threading.Lock()


def _root_key(root: Path) -> str:
    """Normalize a root to compare roots.

    Parameters
    ----------
    root : Path
        Absolute path of a root.

    Returns
    -------
    str
        Path in normal case, so that roots differing only in case on Windows
        are the same.
    """
    return os.path.normcase(str(root))


def get_session(root: Path) -> Session:
    """Get the session of the current process for a root.

    The session is created on first use, its lock is taken, and it is
    closed on normal exit.

    Parameters
    ----------
    root : Path
        Absolute path of an existing root.

    Returns
    -------
    Session
        The session of this process in ``root``.
    """
    key = (os.getpid(), _root_key(root))
    with _sessions_mutex:
        session = _sessions.get(key)
        if session is None:
            session = Session(root)
            _sessions[key] = session
            atexit.register(session.close)
        return session


def register_on_exit(
    root: Path, files: Iterable[Path], directories: Iterable[Path] = ()
) -> None:
    """Register files and directories in the session of this process for a root.

    Unlike ``get_session`` followed by ``Session.register``, a failure
    leaves nothing behind: nothing of this call stays registered, and a
    session this call created is deleted with its session files before the
    error is raised, without being given to anyone else.

    Parameters
    ----------
    root : Path
        Absolute path of an existing root.
    files : Iterable[Path]
        Files to delete on exit, as for ``Session.register``.
    directories : Iterable[Path], default ()
        Directories to delete on exit once empty, as for
        ``Session.register_directories``.
    """
    key = (os.getpid(), _root_key(root))
    with _sessions_mutex:
        session = _sessions.get(key)
        if session is not None:
            session._register(files, directories)
            return
        session = Session(root)
        try:
            session._register(files, directories)
        except BaseException:
            session.discard()
            raise
        _sessions[key] = session
        atexit.register(session.close)


def current_session_id(root: Path) -> str | None:
    """Give the id of the session of this process in a root, if any.

    Parameters
    ----------
    root : Path
        Absolute path of a root.

    Returns
    -------
    str | None
        Id of the session, or ``None`` if this process has none there.
    """
    with _sessions_mutex:
        session = _sessions.get((os.getpid(), _root_key(root)))
    return None if session is None else session.id


def is_session_alive(root: Path, session_id: str) -> bool:
    """Tell whether the process of a session still lives.

    Parameters
    ----------
    root : Path
        Root of the session.
    session_id : str
        Id of the session.

    Returns
    -------
    bool
        Whether its lock is held. A session whose lock file is missing or
        can be locked has ended.
    """
    if session_id == current_session_id(root):
        return True
    fd = lock_file(root / f"{SESSION_PREFIX}{session_id}.lock", blocking=False, create=False)
    if fd is not None:
        unlock_file(fd)
        return False
    return (root / f"{SESSION_PREFIX}{session_id}.lock").exists()


def _is_inside(path: Path, root: Path) -> bool:
    """Tell whether a path is in a root or under it.

    Parameters
    ----------
    path : Path
        Path to check.
    root : Path
        Root directory.

    Returns
    -------
    bool
        Whether ``path`` is ``root`` or below it, compared in normal case.
    """
    a, b = os.path.normcase(str(path)), os.path.normcase(str(root))
    return a != b and os.path.commonpath([a, b]) == b


def collect_stale_sessions(root: Path) -> None:
    """Delete what ended sessions of this user left directly in a root.

    For each session lock file in ``root`` that can be locked, the files
    its JSON file registers and the session files are deleted; if some
    files cannot be deleted, the session files are kept with only those
    files, for a later call. Directories are not deleted, and no other root
    is looked at. Sessions of other
    users, unreadable JSON files, and files that cannot be deleted are
    skipped.

    Parameters
    ----------
    root : Path
        Absolute path of a root.
    """
    try:
        names = [e.name for e in os.scandir(root)]
    except OSError:
        return
    me = user_id()
    for name in names:
        if not (name.startswith(SESSION_PREFIX) and name.endswith(".lock")):
            continue
        session_id = name[len(SESSION_PREFIX) : -len(".lock")]
        if session_id == current_session_id(root):
            continue
        lock_path = root / name
        fd = lock_file(lock_path, blocking=False, create=False)
        if fd is None:
            continue
        json_path = root / f"{SESSION_PREFIX}{session_id}.json"
        data = read_json(json_path)
        if data is None or data.get("marker") != _SESSION_MARKER or data.get("user") != me:
            # A session that is being created has no JSON file yet; one of
            # another user is not ours to collect.
            unlock_file(fd)
            continue
        files = data.get("files")
        registered = [
            f for f in (files if isinstance(files, list) else [])
            if isinstance(f, str) and _is_inside(Path(f), root)
        ]
        delete_paths(root.glob(f"{SESSION_PREFIX}{session_id}.json.*.tmp"))
        left = _delete_registered(registered)
        if left:
            # Keep what could not be deleted registered for a later call.
            with contextlib.suppress(OSError):
                write_json(json_path, {**data, "files": left})
            unlock_file(fd)
            continue
        delete_paths([json_path])
        unlock_and_delete(fd, lock_path)
