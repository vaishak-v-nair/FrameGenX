"""Bounded atomic publication helpers, including Windows sharing violations."""
import logging
from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def directory_lock(root: Path):
    """Exclude concurrent acquisition writers; an OS exit releases ownership."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".acquisition.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise RuntimeError("Another acquisition owns this output directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            unlock()


def atomic_replace(source: str | Path, destination: str | Path):
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.05 * 2 ** attempt)


def remove_temporary(path: str | Path):
    """Cleanup must not hide the original write/download exception."""
    for attempt in range(7):
        try:
            Path(path).unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == 6:
                logging.getLogger(__name__).warning("Temporary-file cleanup deferred: %s", Path(path).name)
                return
            time.sleep(.05 * 2 ** attempt)

