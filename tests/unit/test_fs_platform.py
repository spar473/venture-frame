"""The platform layer for the project library (``fs_platform``) on native Windows.

Most tests here run only on Windows. Each POSIX-only test elsewhere that is skipped on Windows has
its counterpart here, testing the equivalent Windows defence; the pairs are listed in
``docs/development/testing.md``. Every library in this module is a temporary directory.
"""

import json
import os
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from mcda_studio.domain.errors import ProjectExistsError
from mcda_studio.domain.project import Project
from mcda_studio.storage import fs_platform
from mcda_studio.storage import library_backup as lb
from mcda_studio.storage.json_repository import JsonProjectRepository
from mcda_studio.storage.library_lock import (
    LOCK_FILE,
    LibraryBusyError,
    LibraryLockError,
    LibraryOwnership,
)
from mcda_studio.storage.project_store import (
    InvalidTrashEntryError,
    ProjectStore,
    RestoreError,
    UnreadableTrashEntryError,
)

from builders import tiny_project

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="native Windows behaviour")


def deleted(tmp_path: Path) -> tuple[ProjectStore, Project, Path]:
    store = ProjectStore(tmp_path / "library")
    project = tiny_project()
    store.create(project)
    entry = store.delete(project.id)
    return store, project, entry


def junction(link: Path, target: Path) -> None:
    """A directory junction: needs no symlink privilege on Windows."""
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
    )


def directory_link(link: Path, target: Path) -> str:
    """A directory symlink if this account may create one, otherwise a junction."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except OSError:
        junction(link, target)
        return "junction"


def file_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks need the symlink privilege or Developer Mode: {exc}")


# ---- every platform: saved bytes ---------------------------------------------------------------
def test_saved_restored_and_exported_project_bytes_have_no_crlf(tmp_path):
    """Text files are written with newline="\\n"; on Windows this removes the CRLF difference."""
    store, project, entry = deleted(tmp_path)
    assert b"\r\n" not in entry.read_bytes()
    restored = store.restore(entry.name)
    copy = store.restore(entry.name, as_copy=True)
    for stored in (restored, copy):
        data = (store.root / f"{stored.project.id}.json").read_bytes()
        assert b"\n" in data and b"\r\n" not in data
    saved = JsonProjectRepository().save(project, tmp_path / "plain.json")
    assert b"\r\n" not in saved.read_bytes()


# ---- lock ----------------------------------------------------------------------------------------
@windows_only
def test_second_owner_is_refused_while_the_first_holds_the_lock(tmp_path):
    root = tmp_path / "library"
    with LibraryOwnership(root, port=4321) as first:
        assert first.fd is not None
        with pytest.raises(LibraryBusyError, match="4321") as refused, LibraryOwnership(root):
            pass
        assert str(os.getpid()) in str(refused.value)
        assert "do not delete" in str(refused.value)
        # Another process can read the owner diagnostic: the locked byte is not one it reads.
        code = (
            "import json,sys,pathlib;"
            "print(json.loads(pathlib.Path(sys.argv[1]).read_text())['port'])"
        )
        read = subprocess.run(
            [sys.executable, "-c", code, str(root / LOCK_FILE)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert read.returncode == 0, read.stderr
        assert read.stdout.strip() == "4321"
    with LibraryOwnership(root):  # released on close
        pass


@windows_only
def test_lock_byte_lies_beyond_the_diagnostic_and_is_released_on_close(tmp_path):
    root = tmp_path / "library"
    owner = LibraryOwnership(root, port=1)
    owner.__enter__()
    try:
        assert (root / LOCK_FILE).stat().st_size < fs_platform.LOCK_OFFSET
        other = os.open(root / LOCK_FILE, os.O_RDWR | os.O_BINARY)
        try:
            with pytest.raises(BlockingIOError):
                fs_platform.acquire_exclusive(other)
        finally:
            os.close(other)
    finally:
        owner.close()
    again = os.open(root / LOCK_FILE, os.O_RDWR | os.O_BINARY)
    try:
        fs_platform.acquire_exclusive(again)
        fs_platform.release(again)
    finally:
        os.close(again)


@windows_only
@pytest.mark.parametrize("ending", ["exit", "kill"])
def test_lock_is_released_after_the_owning_subprocess_ends(tmp_path, ending):
    root = tmp_path / "library"
    code = (
        "import sys,time\n"
        "from pathlib import Path\n"
        "from mcda_studio.storage.library_lock import LibraryOwnership\n"
        "owner = LibraryOwnership(Path(sys.argv[1]), port=5555)\n"
        "owner.__enter__()\n"
        "print('owned', flush=True)\n"
        "if sys.argv[2] == 'exit':\n"
        "    sys.stdin.readline()\n"
        "    raise SystemExit(0)\n"
        "time.sleep(600)\n"
    )
    # The interpreter itself, not a virtual environment's launcher, so the child is the owner.
    python = getattr(sys, "_base_executable", sys.executable)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    child = subprocess.Popen(
        [python, "-c", code, str(root), ending],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "owned"
        owner = f'"pid": {child.pid}, "port": 5555'
        with pytest.raises(LibraryBusyError, match=owner), LibraryOwnership(root):
            pass
        if ending == "exit":
            child.communicate("\n", timeout=30)  # ends without closing the lock itself
        else:
            child.kill()
            child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
    assert (root / LOCK_FILE).exists()  # the marker survives but owns nothing
    with LibraryOwnership(root):
        pass


@windows_only
def test_symlinked_or_junctioned_lock_marker_is_refused(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    junction(root / LOCK_FILE, target)
    with pytest.raises(LibraryLockError, match="Cannot lock library"), LibraryOwnership(root):
        pass
    assert list(target.iterdir()) == []


# ---- links and reparse points --------------------------------------------------------------------
@windows_only
def test_symlinked_or_junctioned_trash_directory_is_refused(tmp_path):
    store, project, entry = deleted(tmp_path)
    outside = tmp_path / "outside"
    entry.parent.rename(outside)
    kind = directory_link(store.root / ".trash", outside)
    assert fs_platform.is_link(store.root / ".trash"), kind
    with pytest.raises(InvalidTrashEntryError):
        store.list_trash()
    with pytest.raises(InvalidTrashEntryError):
        store.restore(entry.name)
    # Deleting must not move a project through the link into another directory either.
    store.create(project)
    with pytest.raises(InvalidTrashEntryError, match="must not be a symlink"):
        store.delete(project.id)
    assert (store.root / f"{project.id}.json").exists()
    assert sorted(p.name for p in outside.iterdir()) == [entry.name]


@windows_only
def test_symlinked_trash_entry_is_refused(tmp_path):
    store, _, entry = deleted(tmp_path)
    outside = tmp_path / "outside.json"
    entry.rename(outside)
    file_symlink_or_skip(entry, outside)
    before = outside.read_bytes()
    assert not store.list_trash()[0].readable
    with pytest.raises(UnreadableTrashEntryError):
        store.restore(entry.name)
    assert outside.read_bytes() == before


@windows_only
def test_symlinked_project_file_is_not_followed_by_restore(tmp_path):
    store, project, entry = deleted(tmp_path)
    outside = tmp_path / "unrelated.json"
    outside.write_text("keep", encoding="utf-8")
    destination = store.root / f"{project.id}.json"
    file_symlink_or_skip(destination, outside)
    with pytest.raises(ProjectExistsError):
        store.restore(entry.name)
    assert destination.is_symlink() and outside.read_text(encoding="utf-8") == "keep"
    assert entry.exists()


@windows_only
def test_a_directory_in_place_of_a_trash_entry_cannot_block_listing(tmp_path):
    """Counterpart of the POSIX FIFO test: a non-regular entry is listed unreadable and refused."""
    store, _, entry = deleted(tmp_path)
    entry.unlink()
    entry.mkdir()
    assert not store.list_trash()[0].readable
    with pytest.raises(InvalidTrashEntryError, match="not a regular file"):
        store.restore(entry.name)


@windows_only
def test_a_trash_directory_replaced_after_opening_is_detected(tmp_path):
    store, _, entry = deleted(tmp_path)
    handle = fs_platform.open_directory_nofollow(store.root / ".trash")
    moved = tmp_path / "moved"
    (store.root / ".trash").rename(moved)
    junction(store.root / ".trash", moved)
    with pytest.raises(OSError):
        fs_platform.open_read_nofollow(entry.name, handle)
    with pytest.raises(OSError):
        fs_platform.listdir(handle)


@windows_only
def test_backup_skips_a_junction_in_the_library(tmp_path):
    store, _, _ = deleted(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text('{"secret": true}', encoding="utf-8")
    junction(store.root / "linked.json", outside)
    report = lb.create_backup(store.root, tmp_path / "backup.zip")
    assert dict(report.skipped)["linked.json"] == "symlink, not followed"


# ---- restore -------------------------------------------------------------------------------------
@windows_only
def test_restore_refuses_to_overwrite_an_existing_project(tmp_path):
    store, project, entry = deleted(tmp_path)
    active = project.model_copy(update={"title": "Active revision"})
    store.create(active)
    before = (store.root / f"{project.id}.json").read_bytes()
    with pytest.raises(ProjectExistsError):
        store.restore(entry.name)
    assert (store.root / f"{project.id}.json").read_bytes() == before
    assert not list(store.root.glob(".restore-*.tmp"))


@windows_only
def test_restore_as_a_copy_gets_a_new_identity_and_keeps_the_entry(tmp_path):
    store, project, entry = deleted(tmp_path)
    store.create(project.model_copy(update={"title": "Active revision"}))
    copied = store.restore(entry.name, as_copy=True)
    assert copied.project.id != project.id
    assert copied.project.model_copy(update={"id": project.id}) == project
    assert entry.exists() and store.get(copied.project.id) == copied


@windows_only
def test_failure_right_after_publication_retains_recovery_data(tmp_path, monkeypatch):
    """Counterpart of the POSIX directory-sync failure: Windows has no directory fsync, so the
    failure is injected immediately after the no-overwrite link instead."""
    store, project, entry = deleted(tmp_path)
    before = entry.read_bytes()
    real = os.link

    def link_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError("interrupted after publication")

    monkeypatch.setattr(os, "link", link_then_fail)
    with pytest.raises(RestoreError, match="retained"):
        store.restore(entry.name)
    assert entry.read_bytes() == before
    assert store.get(project.id).project == project
    assert not list(store.root.glob(".restore-*.tmp"))


@windows_only
def test_restore_interrupted_after_publication_in_a_subprocess(tmp_path):
    """Counterpart of the POSIX subprocess test that exits during the directory fsync."""
    store, project, entry = deleted(tmp_path)
    before = entry.read_bytes()
    code = (
        "import os,sys\n"
        "from mcda_studio.storage.project_store import ProjectStore\n"
        "real = os.link\n"
        "def link_then_exit(*a, **k):\n"
        "    real(*a, **k)\n"
        "    os._exit(77)\n"
        "os.link = link_then_exit\n"
        "ProjectStore(sys.argv[1]).restore(sys.argv[2])\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(store.root), entry.name], timeout=60)
    assert result.returncode == 77 and entry.read_bytes() == before
    assert store.get(project.id).project == project
    with LibraryOwnership(store.root):
        pass


@windows_only
def test_a_file_system_without_hard_links_is_refused_without_overwriting(tmp_path, monkeypatch):
    store, project, entry = deleted(tmp_path)
    replaced = []
    monkeypatch.setattr(os, "replace", lambda *a, **k: replaced.append(a))

    def no_hard_links(*args, **kwargs):
        raise OSError(0, "Incorrect function", str(args[1]), 1)  # ERROR_INVALID_FUNCTION

    monkeypatch.setattr(os, "link", no_hard_links)
    with pytest.raises(RestoreError, match="NTFS"):
        store.restore(entry.name)
    assert not (store.root / f"{project.id}.json").exists() and replaced == []
    with pytest.raises(fs_platform.HardLinksUnsupportedError):
        fs_platform.link_no_overwrite(entry, tmp_path / "copy.json")


# ---- backup --------------------------------------------------------------------------------------
@windows_only
def test_backup_verify_and_restore_round_trip(tmp_path):
    store, project, entry = deleted(tmp_path)
    kept = project.model_copy(update={"id": "kept", "title": "Kept"})
    store.create(kept)
    archive = tmp_path / "backup.zip"
    report = lb.create_backup(store.root, archive)
    assert (report.active, report.deleted, report.warnings) == (1, 1, False)
    assert lb.verify_backup(archive).active == 1
    target = tmp_path / "restored"
    lb.restore_backup(archive, target)
    for name in (f"{kept.id}.json", f".trash/{entry.name}"):
        assert (target / name).read_bytes() == (store.root / name).read_bytes()
        assert (target / name).stat().st_mtime_ns == (store.root / name).stat().st_mtime_ns
    assert not (target / lb.INCOMPLETE_MARKER).exists()
    assert ProjectStore(target).restore(entry.name).fingerprint == project.fingerprint()
    with pytest.raises(lb.BackupError, match="already exists"):
        lb.restore_backup(archive, target)
    with pytest.raises(lb.BackupError, match="already exists"):
        lb.create_backup(store.root, archive)


@windows_only
def test_new_files_are_created_exclusively_even_over_a_dangling_link(tmp_path):
    """Counterpart of the POSIX O_EXCL | O_NOFOLLOW test. On Windows O_EXCL alone would follow a
    dangling link and create its target, so an existing name of any kind is refused first."""
    (tmp_path / "x.json").write_text("existing", encoding="utf-8")
    handle = fs_platform.open_directory_nofollow(tmp_path)
    with pytest.raises(FileExistsError):
        lb._write_new("x.json", b"new", 0, handle)
    assert (tmp_path / "x.json").read_text(encoding="utf-8") == "existing"
    file_symlink_or_skip(tmp_path / "dangling.json", tmp_path / "nowhere.json")
    with pytest.raises(FileExistsError):
        lb._write_new("dangling.json", b"new", 0, handle)
    assert not (tmp_path / "nowhere.json").exists()


@windows_only
def test_a_backslash_member_name_is_refused_on_windows(tmp_path):
    """Counterpart of the "projects\\a.json" case: zipfile on Windows rewrites a stored backslash
    as "/", so the raw name is checked as well."""
    store, _, _ = deleted(tmp_path)
    archive = tmp_path / "backup.zip"
    lb.create_backup(store.root, archive)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info))
        info = zipfile.ZipInfo("projects/a.json")
        info.filename = "projects\\a.json"  # stored as written, with the backslash
        dst.writestr(info, b"{}")
    with zipfile.ZipFile(bad) as check:
        assert "projects\\a.json" in [i.orig_filename for i in check.infolist()]
    with pytest.raises(lb.BackupError, match="not an allowed member name"):
        lb.verify_backup(bad)


# ---- port ----------------------------------------------------------------------------------------
@windows_only
def test_serve_socket_refuses_a_port_another_socket_holds(tmp_path, monkeypatch):
    """SO_EXCLUSIVEADDRUSE: a port held even with SO_REUSEADDR cannot be taken over, and the
    server's own exclusive socket cannot be joined by a SO_REUSEADDR socket."""
    import uvicorn

    from mcda_studio.api import create_app
    from mcda_studio.startup import run_server

    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    try:
        app = create_app(tmp_path / "library", frontend_dir=False)
        with pytest.raises(ValueError, match="Cannot listen"):
            run_server(app, "127.0.0.1", port, open_browser=False)
    finally:
        holder.close()

    seen: dict[str, int] = {}

    def fake_run(self, sockets=None):
        server_socket = sockets[0]
        seen["exclusive"] = server_socket.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE)
        intruder = socket.socket()
        intruder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            with pytest.raises(OSError):
                intruder.bind(server_socket.getsockname())
        finally:
            intruder.close()
        self.started = True

    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    run_server(create_app(tmp_path / "library", frontend_dir=False), "127.0.0.1", 0, False)
    assert seen["exclusive"] == 1


@windows_only
def test_lock_marker_diagnostic_is_plain_json_without_crlf(tmp_path):
    root = tmp_path / "library"
    with LibraryOwnership(root, port=8765):
        data = (root / LOCK_FILE).read_bytes()
    assert b"\r" not in data and json.loads(data) == {"pid": os.getpid(), "port": 8765}
