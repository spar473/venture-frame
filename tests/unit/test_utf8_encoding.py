"""Every text read and write names its encoding, so nothing depends on the platform's locale.

On Windows the locale's code page decides the default (cp1252, cp949, ...), so a UTF-8 file read
with ``Path.read_text()`` is decoded wrongly or refused. Ruff's ``PLW1514`` needs preview mode and
misses ``read_text``/``write_text`` on expressions it cannot type, so this test walks the syntax
tree instead and fails on any call that would use the locale:

- ``open(...)`` and ``Path.open(...)`` in text mode without ``encoding``;
- ``.read_text()`` and ``.write_text(...)`` without ``encoding``;
- ``subprocess`` calls with ``text=True`` or ``universal_newlines=True`` without ``encoding``;
- ``tempfile`` text-mode files without ``encoding``.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("src", "tests", "scripts", "examples", "hatch_build.py")
SUBPROCESS = {"run", "check_output", "check_call", "call", "Popen"}
TEMPFILE = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _has(call: ast.Call, name: str) -> bool:
    return any(k.arg == name or k.arg is None for k in call.keywords)  # **kwargs counts


def _text_mode(mode: ast.expr | None, default_text: bool) -> bool:
    if mode is None:
        return default_text
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return "b" not in mode.value
    return False  # a computed mode: cannot tell, not flagged


def _true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def problems(source: str, name: str = "<string>") -> list[str]:
    """``file:line: call`` for every locale-dependent text call in ``source``."""
    out: list[str] = []
    for node in ast.walk(ast.parse(source, name)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        where = f"{name}:{node.lineno}"
        if isinstance(func, ast.Name) and func.id == "open":
            mode = node.args[1] if len(node.args) > 1 else _keyword(node, "mode")
            if _text_mode(mode, True) and len(node.args) < 4 and not _has(node, "encoding"):
                out.append(f"{where}: open() in text mode without encoding")
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            if attr == "read_text" and not node.args and not _has(node, "encoding"):
                out.append(f"{where}: read_text() without encoding")
            elif attr == "write_text" and len(node.args) < 2 and not _has(node, "encoding"):
                out.append(f"{where}: write_text() without encoding")
            elif attr == "open":
                first = node.args[0] if node.args else _keyword(node, "mode")
                path_open = first is None or (
                    isinstance(first, ast.Constant) and isinstance(first.value, str)
                )
                if (
                    path_open
                    and _text_mode(first, True)
                    and len(node.args) < 3
                    and not _has(node, "encoding")
                ):
                    out.append(f"{where}: .open() in text mode without encoding")
            elif attr in SUBPROCESS and isinstance(func.value, ast.Name):
                text_mode = _true(_keyword(node, "text")) or _true(
                    _keyword(node, "universal_newlines")
                )
                if func.value.id == "subprocess" and text_mode and not _has(node, "encoding"):
                    out.append(f"{where}: subprocess text pipes without encoding")
            elif attr in TEMPFILE and isinstance(func.value, ast.Name):
                mode = node.args[0] if node.args else _keyword(node, "mode")
                if (
                    func.value.id == "tempfile"
                    and _text_mode(mode, False)
                    and not _has(node, "encoding")
                ):
                    out.append(f"{where}: tempfile text mode without encoding")
    return out


def scanned_files() -> list[Path]:
    files: list[Path] = []
    for item in SCANNED:
        path = ROOT / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                p
                for p in path.rglob("*.py")
                if not any(part in {"node_modules", ".venv", "__pycache__"} for part in p.parts)
            )
    return sorted(files)


def test_every_text_read_and_write_names_its_encoding() -> None:
    found = []
    for path in scanned_files():
        found += problems(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
    assert not found, "locale-dependent text I/O:\n" + "\n".join(found)


def test_the_checker_catches_each_kind_and_accepts_the_fixed_forms() -> None:
    bad = """
import subprocess, tempfile
open("a.txt")
open("a.txt", "w")
p.read_text()
p.write_text("x")
p.open()
p.open("w", newline="")
subprocess.run(["x"], text=True)
subprocess.check_output(["x"], universal_newlines=True)
tempfile.NamedTemporaryFile("w")
"""
    assert len(problems(bad)) == 9
    good = """
import subprocess, tempfile
open("a.bin", "rb")
open("a.txt", encoding="utf-8")
open("a.txt", "w", -1, "utf-8")
p.read_text(encoding="utf-8")
p.read_text("utf-8")
p.write_text("x", encoding="utf-8")
p.write_text("x", "utf-8")
p.open("rb")
p.open("w", encoding="utf-8", newline="")
zf.open(info)
tarfile.open(path)
subprocess.run(["x"], text=True, encoding="utf-8")
subprocess.run(["x"], capture_output=True)
tempfile.NamedTemporaryFile()
tempfile.NamedTemporaryFile("w", encoding="utf-8")
f(**options)
"""
    assert problems(good) == []
