# Changelog

## Unreleased

### Native Windows

- The engine, command line, `mcda serve` and the project library run on native Windows. The
  "macOS/Linux only" refusal of library ownership is removed.
- Platform-specific file operations are in one module, `mcda_studio.storage.fs_platform`. On
  macOS and Linux it makes the same system calls as before. On Windows it uses a `msvcrt`
  byte-range lock and link, reparse-point and identity checks in place of `flock`, `O_NOFOLLOW`
  and `dir_fd`; the differences are listed in
  [known limitations](docs/development/known-limitations.md).
- `mcda serve` binds with `SO_EXCLUSIVEADDRUSE` on Windows, so no other process can take its port.
- Every text file is written with LF line endings on every platform, and `.gitattributes` checks
  text files out with LF, so a Windows checkout and its saved projects match the POSIX bytes.
- `scripts/build_release.py` checks wheel member names with `/` and finds `npm.cmd`, so both
  build routes run on Windows. `scripts/smoke_install.py` understands the Windows
  virtual-environment layout and Playwright cache, and pins pipx to its pip backend.
- The hash locks install on Windows: each gains `colorama==0.4.6` (pip-compile's Windows output)
  behind a `sys_platform == "win32"` marker, so macOS and Linux install exactly what they did.
- CI gains a `windows-python312` job (Python and frontend gates, serve smoke, installed smoke over
  HTTP and the CLI) and a job comparing the Linux and Windows byte-identity manifests.

## 0.1.0 — 17 November 2026

First public release of Venture Frame: the MCDA Studio engine, command line and local interface,
with four venture case studies in [`cases/`](cases/README.md).
