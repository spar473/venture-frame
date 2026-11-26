"""The command line never crashes on a console encoding that cannot show a name, and its
machine-readable output stays UTF-8 (``mcda_studio/console_streams.py``).

The subprocess tests run the real ``python -m mcda_studio`` with UTF-8 mode off, no
``PYTHONIOENCODING`` and an ASCII ``C`` locale (on Linux that is a 7-bit console; on Windows the
ANSI code page, cp949 on the machine this was written on, decides). They also force narrow
encodings through ``PYTHONIOENCODING``.
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mcda_studio.console_streams import configure_stream

ROOT = Path(__file__).resolve().parents[2]
NAME = "Māori Kaitiaki – grade ≥ 3"


class Stream(io.TextIOWrapper):
    def __init__(self, encoding: str, tty: bool) -> None:
        super().__init__(io.BytesIO(), encoding=encoding)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def written(self) -> bytes:
        self.flush()
        return self.buffer.getvalue()  # type: ignore[attr-defined]


def test_a_pipe_in_a_narrow_encoding_becomes_utf8() -> None:
    stream = Stream("cp949", tty=False)
    assert configure_stream(stream, user_chose_encoding=False) == "utf-8"
    stream.write(NAME)
    assert stream.written().decode("utf-8") == NAME


def test_a_narrow_terminal_keeps_its_encoding_and_escapes_what_it_cannot_show() -> None:
    stream = Stream("ascii", tty=True)
    assert configure_stream(stream, user_chose_encoding=False) == "backslashreplace"
    stream.write("Māori ≥")
    assert stream.written() == b"M\\u0101ori \\u2265"


def test_an_encoding_the_user_chose_is_kept_without_crashing() -> None:
    stream = Stream("cp949", tty=False)
    assert configure_stream(stream, user_chose_encoding=True) == "backslashreplace"
    stream.write("ū")
    assert stream.written() == b"\\u016b"


def test_utf8_streams_and_streams_that_cannot_be_reconfigured_are_left_alone() -> None:
    assert configure_stream(Stream("utf-8", tty=False), user_chose_encoding=False) is None
    assert configure_stream(Stream("UTF8", tty=True), user_chose_encoding=True) is None
    assert configure_stream(object(), user_chose_encoding=False) is None


# ---- the real command line ------------------------------------------------------------------


@pytest.fixture(scope="module")
def project(tmp_path_factory: Any) -> Path:
    raw = json.loads(
        (ROOT / "examples" / "supplier_qualification" / "project.json").read_text(encoding="utf-8")
    )
    raw["alternatives"][0]["name"] = NAME
    raw["title"] = f"{raw['title']} – {NAME}"
    path = tmp_path_factory.mktemp("encoding") / "project.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def narrow_env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONIOENCODING", "PYTHONUTF8", "LC_ALL", "LANG", "LC_CTYPE"}
    }
    # UTF-8 mode off and no locale coercion, so "C" really is ASCII on Linux.
    env.update(PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C")
    env["PYTHONPATH"] = str(ROOT / "src")
    env.update(extra)
    return env


def mcda(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "mcda_studio", *args], capture_output=True, env=env, cwd=ROOT
    )


def test_piped_output_is_utf8_in_an_ascii_locale(project: Path) -> None:
    env = narrow_env()
    table = mcda("evaluate", str(project), env=env)
    assert table.returncode == 0, table.stderr.decode("utf-8", "replace")
    assert "Māori" in table.stdout.decode("utf-8")
    full = mcda("evaluate", str(project), "--json", env=env)
    assert full.returncode == 0, full.stderr.decode("utf-8", "replace")
    names = [row["name"] for row in json.loads(full.stdout.decode("utf-8"))["ranking"]]
    assert NAME in names
    check = mcda("validate", str(project), env=env)
    assert check.returncode == 0, check.stderr.decode("utf-8", "replace")


def test_files_are_utf8_json_whatever_the_console(project: Path, tmp_path: Path) -> None:
    encodings = ["ascii"] + (["cp949"] if sys.platform == "win32" else [])
    for encoding in encodings:
        out = tmp_path / f"result-{encoding}.json"
        result = mcda(
            "evaluate", str(project), "-o", str(out), env=narrow_env(PYTHONIOENCODING=encoding)
        )
        assert result.returncode == 0, (encoding, result.stderr.decode(encoding, "replace"))
        result.stdout.decode(encoding)  # what the console got is valid in its own encoding
        saved = json.loads(out.read_bytes().decode("utf-8"))
        assert NAME in [row["name"] for row in saved["ranking"]]


def test_a_forced_narrow_console_escapes_instead_of_crashing(project: Path) -> None:
    for encoding in ["ascii"] + (["cp949"] if sys.platform == "win32" else []):
        result = mcda("evaluate", str(project), env=narrow_env(PYTHONIOENCODING=encoding))
        assert result.returncode == 0, (encoding, result.stderr.decode(encoding, "replace"))
        text = result.stdout.decode(encoding)
        assert "Kaitiaki" in text
