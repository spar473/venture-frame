"""Hooks shared by every test directory."""

import sys

import pytest

if sys.platform == "win32":
    ERROR_PRIVILEGE_NOT_HELD = 1314

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
        """Windows lets only privileged accounts, or Developer Mode, create symbolic links. A test
        that needs one is reported as skipped, with the reason, rather than as an error; with the
        privilege (as on the CI runner and in Developer Mode) every such test runs."""
        outcome = yield
        report = outcome.get_result()
        error = call.excinfo.value if call.excinfo is not None else None
        if isinstance(error, OSError) and getattr(error, "winerror", None) == (
            ERROR_PRIVILEGE_NOT_HELD
        ):
            report.outcome = "skipped"
            report.longrepr = (
                str(item.path),
                item.location[1] or 0,
                "Skipped: creating a symbolic link needs the symlink privilege or Developer Mode",
            )
