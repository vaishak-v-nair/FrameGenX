"""One offline, process-safe lease for FrameGenX CUDA work.

No torch import or CUDA query is allowed here. A single host-user lock is
deliberately conservative: visible-device aliases, separate checkouts and API
workers must not each believe they own this laptop's GPU. Other applications
and other operating-system users do not participate in this cooperative gate.
"""
from contextlib import contextmanager
from functools import wraps
import gc
import os
from pathlib import Path
import tempfile
import threading
import time


class DeviceBusyError(RuntimeError):
    code = "device_busy"

    def __init__(self):
        super().__init__("Another FrameGenX operation owns the device. Retry after it finishes.")


class DeviceLeaseRequired(RuntimeError):
    pass


_mutex = threading.RLock()
_state = {"pid": os.getpid(), "thread": None, "depth": 0, "file": None, "poisoned": False}


def _lock_path() -> Path:
    # Not rooted in a checkout or CUDA ordinal; both can refer to the same GPU.
    root = Path(tempfile.gettempdir()) / "framegenx-device"
    root.mkdir(parents=True, exist_ok=True)
    return root / "cuda.lock"


def _os_lock(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _os_unlock(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _after_fork():
    # A forked child is a separate owner, never a nested call of its parent.
    global _mutex
    inherited = _state["file"]
    if inherited is not None:
        inherited.close()  # do not unlock the parent's inherited flock
    _state.update(pid=os.getpid(), thread=None, depth=0, file=None, poisoned=False)
    _mutex = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def owns_device() -> bool:
    with _mutex:
        return (_state["pid"] == os.getpid() and not _state["poisoned"] and
                _state["thread"] == threading.get_ident() and _state["depth"] > 0)


def require_device_lease():
    if not owns_device():
        raise DeviceLeaseRequired("CUDA work requires 'with device_lease()' before probing or loading models.")


def _acquire():
    with _mutex:
        thread = threading.get_ident()
        if _state["poisoned"]:
            raise DeviceBusyError()
        if _state["depth"]:
            if _state["thread"] != thread:
                raise DeviceBusyError()
            _state["depth"] += 1
            return
        # a+b preserves the permanent lock inode. Never unlink a lock file.
        stream = _lock_path().open("a+b")
        try:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            # Windows can briefly retain a dying process's byte-range lock
            # after its process handle is signaled. Bound this grace period;
            # a live owner still produces a prompt, explicit busy outcome.
            deadline = time.monotonic() + .25
            while True:
                try:
                    _os_lock(stream)
                    break
                except OSError as error:
                    import errno
                    if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} or time.monotonic() >= deadline:
                        raise
                    time.sleep(.025)
        except OSError as error:
            stream.close()
            import errno
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise DeviceBusyError() from None
            raise
        _state.update(pid=os.getpid(), thread=thread, depth=1, file=stream)


def _release():
    with _mutex:
        if _state["poisoned"]:
            # A failed cleanup cannot advertise free hardware. OS exit releases
            # the handle; this process refuses further work until restart.
            return
        require_device_lease()
        _state["depth"] -= 1
        if _state["depth"] == 0:
            stream = _state["file"]
            try:
                _os_unlock(stream)
            finally:
                stream.close()
                _state.update(thread=None, file=None)


@contextmanager
def device_lease():
    """Fail promptly on contention; nested calls in one thread share ownership."""
    _acquire()
    try:
        yield
    finally:
        _release()


def exclusive_device(function=None, *, cleanup=None):
    """Guard a synchronous entrypoint, cleaning owned resources before release."""
    def decorate(func):
        @wraps(func)
        def guarded(*args, **kwargs):
            with device_lease():
                try:
                    return func(*args, **kwargs)
                finally:
                    try:
                        if cleanup:
                            getattr(args[0], cleanup)()
                        gc.collect()
                        # Import torch only if the operation already imported it.
                        import sys
                        torch = sys.modules.get("torch")
                        if torch is not None and torch.cuda.is_initialized():
                            torch.cuda.empty_cache()
                    except BaseException:
                        with _mutex:
                            _state["poisoned"] = True
                        raise
        return guarded
    return decorate(function) if function is not None else decorate
