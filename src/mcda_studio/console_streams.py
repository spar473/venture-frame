"""Make the command line's standard streams safe on any platform's console encoding.

Python encodes ``sys.stdout`` and ``sys.stderr`` with the locale's encoding when they are pipes or
files (on Windows, the ANSI code page: cp1252, cp949, ...), and with ``errors="strict"``. A label
such as "Māori", an en dash or "≥" then crashes a command that only prints it.

The rule, applied once per command by the ``mcda`` entry point (the Typer callback in ``cli.py``):

- **Already UTF-8:** left alone.
- **Redirected or piped** (not a terminal) and ``PYTHONIOENCODING`` not set: switched to UTF-8,
  so machine-readable output (JSON, CSV on standard output) is UTF-8 everywhere, as files written
  with ``--output`` already are.
- **A terminal that cannot show UTF-8**, or an encoding the user chose with
  ``PYTHONIOENCODING``: the encoding is kept and characters it cannot show are written as
  backslash escapes (``errors="backslashreplace"``) instead of crashing.

Files are never affected: every file the application writes is opened with ``encoding="utf-8"``.
"""

from __future__ import annotations

import codecs
import os
import sys
from typing import Any

FALLBACK_ERRORS = "backslashreplace"


def _is_utf8(encoding: str | None) -> bool:
    if not encoding:
        return False
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def configure_stream(stream: Any, *, user_chose_encoding: bool) -> str | None:
    """Reconfigure one text stream by the rule above; returns what was done, or None."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None or _is_utf8(getattr(stream, "encoding", None)):
        return None
    try:
        interactive = bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        interactive = False
    if interactive or user_chose_encoding:
        reconfigure(errors=FALLBACK_ERRORS)
        return FALLBACK_ERRORS
    reconfigure(encoding="utf-8")
    return "utf-8"


def configure_streams() -> None:
    """Apply the rule to standard output and standard error."""
    chose = bool(os.environ.get("PYTHONIOENCODING"))
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            configure_stream(stream, user_chose_encoding=chose)
