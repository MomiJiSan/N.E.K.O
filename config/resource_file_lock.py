"""Standard-library file locks shared by configuration and resource delivery."""

from contextlib import contextmanager
import os
from pathlib import Path
import stat


class ResourceFileLockBusy(OSError):
    pass


@contextmanager
def resource_file_lock(path: Path):
    """The OS releases this lock when a worker is killed or the process exits."""
    if path.is_symlink():
        raise OSError("unsafe resource lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        if (not stat.S_ISREG(os.fstat(handle.fileno()).st_mode)
                or os.fstat(handle.fileno()).st_size > 1):
            raise OSError("unsafe resource lock")
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ResourceFileLockBusy("resource operation busy") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
