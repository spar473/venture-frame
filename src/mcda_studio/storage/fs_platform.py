"""Platform-specific file operations for the project library, behind one small interface.

On POSIX (macOS, Linux) every function is the system call the library has always used: ``flock``,
``O_NOFOLLOW``, ``dir_fd``-relative opens and links, and directory ``fsync``. Nothing here changes
their behaviour or their errors.

Native Windows has no ``flock``, ``O_NOFOLLOW``, ``dir_fd`` or directory ``fsync``. The Windows
functions keep the same guarantees where Windows can give them and check where it cannot:

* **Lock**: ``msvcrt.locking`` on one reserved byte far beyond the diagnostic text, so another
  process can still read who owns the library. Windows byte-range locks are mandatory, so the
  locked byte is never one anybody needs to read. Closing the descriptor, or the process ending,
  releases it.
* **Links and reparse points**: before opening, ``os.lstat`` refuses a symbolic link or anything
  with ``FILE_ATTRIBUTE_REPARSE_POINT`` (junctions included). After opening, the descriptor's
  identity (``st_ino``, ``st_dev``) must equal the path's, and the path must still not be a link.
  A small check-then-open window remains; see ``docs/development/known-limitations.md``.
* **Directories** are a checked path (:class:`DirHandle`) instead of a descriptor. Each operation
  inside one re-checks that the path is still the same directory and not a link or reparse point.
* **No-overwrite publication** is ``os.link``, which refuses any existing name, links included.
  A file system without hard links (FAT, exFAT, some network or synced folders) is refused with
  :class:`HardLinksUnsupportedError`; there is never a fallback to ``os.replace``.
* **Directory fsync** is a documented no-op: Windows cannot ``fsync`` a directory.
* Every ``os.open`` adds ``O_BINARY``, because Windows descriptors otherwise translate newlines.
"""

import errno
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

WINDOWS = sys.platform == "win32"
#: Windows: the byte that carries the lock, far beyond any diagnostic text.
LOCK_OFFSET = 1 << 20
#: The smallest step between two modified times that the library's file system keeps apart, in
#: nanoseconds: NTFS stores 100 ns ticks; APFS and ext4 store nanoseconds. A coarser file system
#: (FAT, exFAT, HFS+) keeps fewer apart; see ``docs/development/known-limitations.md``.
MTIME_STEP_NS = 100 if WINDOWS else 1


class NotRegularFileError(OSError):
    """Windows only: the path names a directory or another non-regular file."""


class HardLinksUnsupportedError(OSError):
    """Windows only: this file system cannot publish a file without risking an overwrite."""


if sys.platform == "win32":

    @dataclass(frozen=True)
    class DirHandle:
        """A directory that was checked when opened: not a link or reparse point."""

        path: str
        ino: int
        dev: int

    def _is_link_stat(info: os.stat_result) -> bool:
        return stat.S_ISLNK(info.st_mode) or bool(
            info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )

    def _refuse_link(path: str) -> os.stat_result | None:
        """The path's ``lstat``, or None if nothing is there; refuses a link or reparse point."""
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return None
        if _is_link_stat(info):
            raise OSError(errno.ELOOP, "Refusing a symbolic link or reparse point", path)
        return info

    def _same(first: os.stat_result, second: os.stat_result) -> bool:
        return (first.st_ino, first.st_dev) == (second.st_ino, second.st_dev)

    def _checked(directory: "DirHandle") -> str:
        """The directory's path, after checking it is still the directory that was opened."""
        current = _refuse_link(directory.path)
        if current is None or not stat.S_ISDIR(current.st_mode):
            raise OSError(errno.ENOENT, "Directory no longer exists", directory.path)
        if (current.st_ino, current.st_dev) != (directory.ino, directory.dev):
            raise OSError(errno.ESTALE, "Directory was replaced while in use", directory.path)
        return directory.path

    def _child(name: str, directory: "DirHandle | None") -> str:
        return name if directory is None else os.path.join(_checked(directory), name)

    def _open_checked(path: str, flags: int, mode: int = 0o777) -> int:
        """Open with the Windows link defence: lstat first, then compare identities."""
        if flags & os.O_EXCL and os.path.lexists(path):
            # O_EXCL alone would follow a dangling link on Windows and create its target.
            raise FileExistsError(errno.EEXIST, "File exists", path)
        before = _refuse_link(path)
        if before is not None and not stat.S_ISREG(before.st_mode):
            raise NotRegularFileError(errno.EISDIR, "Not a regular file", path)
        fd = os.open(path, flags | os.O_BINARY | os.O_NOINHERIT, mode)
        try:
            opened = os.fstat(fd)
            after = _refuse_link(path)
            if after is None or not _same(after, opened):
                raise OSError(errno.ELOOP, "Path changed while it was opened", path)
            if before is not None and not _same(before, opened):
                raise OSError(errno.ELOOP, "Path changed while it was opened", path)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _hard_link(source: str, target: str) -> None:
        try:
            os.link(source, target)
        except FileExistsError:
            raise
        except OSError as exc:
            # ERROR_INVALID_FUNCTION / ERROR_NOT_SUPPORTED: FAT, exFAT and many network shares.
            if getattr(exc, "winerror", None) in (1, 50):
                raise HardLinksUnsupportedError(
                    exc.errno,
                    "This folder's file system has no hard links, which the library needs to "
                    "publish a file without overwriting another; use a folder on an NTFS drive",
                    target,
                ) from exc
            raise

    # ---- lock ---------------------------------------------------------------------------------
    def open_lock_file(path: Path) -> int:
        return _open_checked(str(path), os.O_RDWR | os.O_CREAT, 0o600)

    def acquire_exclusive(fd: int) -> None:
        """Lock without waiting; raises BlockingIOError if another descriptor holds the lock."""
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLOCK):
                raise BlockingIOError(errno.EAGAIN, "Library lock is held elsewhere") from None
            raise
        finally:
            os.lseek(fd, 0, os.SEEK_SET)

    def release(fd: int) -> None:
        """Unlock before closing, so the next owner never waits for the system to notice."""
        try:
            os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # not locked by this descriptor; closing releases anything left

    # ---- links --------------------------------------------------------------------------------
    def is_link(item: "os.DirEntry[str] | Path") -> bool:
        """A symbolic link, junction or any other reparse point."""
        try:
            info = item.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        return _is_link_stat(info)

    # ---- directories --------------------------------------------------------------------------
    def open_directory(path: str | Path) -> DirHandle:
        """A directory reached by following links (its resolved location is then checked)."""
        return open_directory_nofollow(os.path.realpath(path))

    def open_directory_nofollow(name: str | Path, directory: DirHandle | None = None) -> DirHandle:
        path = _child(str(name), directory)
        info = _refuse_link(path)
        if info is None:
            raise FileNotFoundError(errno.ENOENT, "No such directory", path)
        if not stat.S_ISDIR(info.st_mode):
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory", path)
        return DirHandle(path, info.st_ino, info.st_dev)

    def close_directory(directory: DirHandle) -> None:
        """Nothing to release: a Windows directory handle here is a checked path."""

    def scan(directory: DirHandle) -> list[os.DirEntry[str]]:
        with os.scandir(_checked(directory)) as entries:
            return list(entries)

    def listdir(directory: DirHandle) -> list[str]:
        return os.listdir(_checked(directory))

    def mkdir_in(name: str, directory: DirHandle) -> None:
        os.mkdir(_child(name, directory))

    def unlink_in(name: str, directory: DirHandle) -> None:
        path = _child(name, directory)
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", path)
        os.unlink(path)

    def rmdir_in(name: str, directory: DirHandle) -> None:
        os.rmdir(_child(name, directory))

    def fsync_directory(directory: DirHandle) -> None:
        """Documented no-op: Windows cannot fsync a directory (NTFS journals its metadata)."""

    # ---- files --------------------------------------------------------------------------------
    def open_read_nofollow(name: str, directory: DirHandle) -> int:
        return _open_checked(_child(name, directory), os.O_RDONLY)

    def create_exclusive(name: str, directory: DirHandle, mode: int) -> int:
        """A new file; refuses any existing name, a dangling link included."""
        return _open_checked(_child(name, directory), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)

    def set_times(fd: int, name: str, directory: DirHandle, modified_ns: int) -> None:
        path = _child(name, directory)
        os.utime(path, ns=(modified_ns, modified_ns))
        after = _refuse_link(path)
        if after is None or not _same(after, os.fstat(fd)):
            raise OSError(errno.ELOOP, "Path changed while its times were set", path)

    def publish_no_overwrite(staged: Path, destination: Path, directory: DirHandle) -> None:
        """Atomic publication that refuses any existing entry, including a link."""
        _hard_link(_child(staged.name, directory), _child(destination.name, directory))

    def link_no_overwrite(source: Path, target: Path, *, follow_symlinks: bool = True) -> None:
        _hard_link(str(source), str(target))

else:
    DirHandle = int  # a directory descriptor

    # ---- lock ---------------------------------------------------------------------------------
    def open_lock_file(path: Path) -> int:
        return os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)

    def acquire_exclusive(fd: int) -> None:
        """Lock without waiting; raises BlockingIOError if another descriptor holds the lock."""
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(fd: int) -> None:
        """Closing releases flock, including after ordinary process termination."""

    # ---- links --------------------------------------------------------------------------------
    def is_link(item: "os.DirEntry[str] | Path") -> bool:
        return item.is_symlink()

    # ---- directories --------------------------------------------------------------------------
    def open_directory(path: str | Path) -> DirHandle:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY)

    def open_directory_nofollow(name: str | Path, directory: DirHandle | None = None) -> DirHandle:
        if directory is None:
            return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)

    def close_directory(directory: DirHandle) -> None:
        os.close(directory)

    def scan(directory: DirHandle) -> list[os.DirEntry[str]]:
        with os.scandir(directory) as entries:
            return list(entries)

    def listdir(directory: DirHandle) -> list[str]:
        return os.listdir(directory)

    def mkdir_in(name: str, directory: DirHandle) -> None:
        os.mkdir(name, dir_fd=directory)

    def unlink_in(name: str, directory: DirHandle) -> None:
        os.unlink(name, dir_fd=directory)

    def rmdir_in(name: str, directory: DirHandle) -> None:
        os.rmdir(name, dir_fd=directory)

    def fsync_directory(directory: DirHandle) -> None:
        os.fsync(directory)

    # ---- files --------------------------------------------------------------------------------
    def open_read_nofollow(name: str, directory: DirHandle) -> int:
        return os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)

    def create_exclusive(name: str, directory: DirHandle, mode: int) -> int:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        return os.open(name, flags, mode, dir_fd=directory)

    def set_times(fd: int, name: str, directory: DirHandle, modified_ns: int) -> None:
        os.utime(fd, ns=(modified_ns, modified_ns))

    def publish_no_overwrite(staged: Path, destination: Path, directory: DirHandle) -> None:
        """Atomic publication that refuses any existing entry, including a symlink."""
        root_fd = directory
        os.link(
            staged.name,
            destination.name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
            follow_symlinks=False,
        )

    def link_no_overwrite(source: Path, target: Path, *, follow_symlinks: bool = True) -> None:
        if follow_symlinks:
            os.link(source, target)
        else:
            os.link(source, target, follow_symlinks=False)
