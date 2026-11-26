# Changelog

## Unreleased

### Decision confidence

- Each robustness-band row gains `first`, `top_ranks` and `classes`: the share of SMAA-2 draws in
  which the alternative is first (ties included), in the top `top_k` ranks (`min(3, alternatives)`)
  and in each class, with SMAA-2's exact Clopper–Pearson interval at its `confidence` level, or
  exact probabilities under enumeration. The response adds `top_k`, `confidence` and
  `confidence_label`. Read from SMAA-2's acceptabilities; nothing is sampled again and no method
  output changes.
- The screening summary shows a Decision confidence table under the summary table, and the HTML
  screening report gains a "Decision confidence by profile" section. The venture study's golden
  report is regenerated (owner-approved); the supplier report is unchanged.
- Shares are labelled as shares of draws under the stated uncertainty, never as probabilities of
  success. Design: [robustness bands](docs/design/robustness-bands.md#decision-confidence).

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
- The typed-proposal runner spaces requests from the end of the previous one (so a slow local
  connection on Windows cannot bring two requests closer than the interval) and retries a refused
  rename of its snapshot index briefly on Windows.
- The hash locks install on Windows: each gains `colorama==0.4.6` (pip-compile's Windows output)
  behind a `sys_platform == "win32"` marker, so macOS and Linux install exactly what they did.
- CI gains a `windows-python312` job (Python and frontend gates, serve smoke, installed smoke over
  HTTP and the CLI) and a job comparing the Linux and Windows byte-identity manifests.
- The Linux CI job's time limit rises from 25 to 35 minutes, the same as the Windows job. It takes
  about 18–24 minutes, and the 25-minute limit cancelled passing runs on slower runners.

### Explicit cross-site check

- Every `POST`, `PUT`, `PATCH` and `DELETE` now checks `Origin` and `Sec-Fetch-Site`. A request
  from a page on another site is refused with HTTP 403 and the new error code
  `CROSS_ORIGIN_REFUSED`. Before, such a write was refused only indirectly, because the API accepts
  JSON only and browsers preflight cross-origin JSON. Requests without these browser headers (the
  CLI, scripts, `curl`) are unaffected, and so are `GET`, `HEAD` and `OPTIONS`.
- The development server (`npm run dev`) presents proxied API requests with the backend's origin,
  so the guard accepts them.

### Partner view

- **Venture Frame in the interface.** The page title, wordmark and connection error name Venture
  Frame, from one frontend product configuration (`frontend/src/product.ts`). The Python package
  `mcda_studio`, the CLI `mcda`, the API title, Python's error messages and the backup manifest
  are unchanged.
- **Start page.** "Screen companies for due diligence", with three starters: Blank, Venture
  screening and the evidence-and-model workflows. Every other starter stays in **New project… →
  Starter gallery**, linked under the cards.
- **Analyst view.** The header leads with **Partner weighting**; Method, Set up and Scenario sit
  in a collapsed **Advanced** section whose label still names a non-default method or scenario.
  Values, margins and contributions in the screening summary, company detail and partner
  comparison show three decimals ("< 0.001" below that), with the full value on hover. The closest
  change reads "Moves to Monitor if Problem severity and willingness to pay gains 6.1 pts of
  weight". The Builder view, every export, the screening report and all JSON and CSV keep their
  exact text and precision.
- **Open-question counts** say whether they come from the model only or include the evidence
  dossier, and offer to load the study's evidence. How questions are counted is unchanged.
- **Robustness bands** (new read-only output; [design](docs/design/robustness-bands.md)). From
  SMAA-2's rank acceptabilities under the uncertainty a project states: each alternative's rank
  range (the shortest run of ranks holding at least 80 % of draws) and the groups whose ranges
  overlap, shown as "Too close to separate". `POST /api/screening-summary/robustness` and the
  stored-project route, behind the cross-site check. The screening summary shows `2 (2–6)`, a
  group marker and one lead sentence; the bands load after the summary and are hidden during a
  what-if trial. SMAA-2, the screening summary response and its fixture are unchanged.
- **Screening report** (version 3) adds each rank range and one line per too-close group, or one
  line saying why there are none. The case reports, their previews and the two golden reports are
  regenerated; nothing else in them changed.
- Development: `@axe-core/playwright` (MPL-2.0, development only) for an accessibility scan in the
  new browser spec;
  `scripts/mutation_check.py` gains nine robustness mutants and two report mutants, and finds
  `npm.cmd`/`npx.cmd` on Windows.

## 0.1.0 — 17 November 2026

First public release of Venture Frame: the MCDA Studio engine, command line and local interface,
with four venture case studies in [`cases/`](cases/README.md).
