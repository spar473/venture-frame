# Changelog

## Unreleased

### The true conviction prior and samples in SMAA-2 (schema 8)

- **The conviction prior.** `mcda calibrate prior` draws a profile's weights from Dirichlet(κ·w0)
  per sibling group, with κ moment-matched to the DD3 conviction (555 / 136 / 47 on case 2's top
  level) or stated. It writes a `weight-samples` sidecar with the new `basis: "prior"`.
- **SMAA-2's `samples` weight source.** The new parameters `weight_source`, `weight_samples_sha256`
  and `resample` let a run read its weights from a sidecar.
  - S ≤ N uses distinct vectors; S > N is allowed only with `resample`, and every output says so.
  - Every refusal has a code; the result records the source.
  - A saved samples configuration is schema 8. Every other project keeps its version and bytes.
  - `mcda evaluate --weight-samples`; `POST /api/evaluate` takes `weight_samples`.
- **The Judgement tab.** The decision-confidence comparison gains a true-conviction column with
  the κ used, and a "Weight uncertainty" choice.
- **Byte identity.** All 969 existing outputs are byte-identical. Design:
  [sampled weights in SMAA-2](docs/design/sampled-weights-in-smaa.md).

### Reference rating (deep dive 6)

- A new opt-in sidecar, `rating-proposals` version 1: proposed labels for reference cases, each
  with one dated public source, a locator and an excerpt of at most 25 words. "Not enough evidence"
  is a proposal of its own and leaves a criterion unrated.
- `mcda reference-cases packets` writes per-case packets without outcome information (context rows
  only from a whitelist). `review-sheet` writes a UTF-8 CSV with a byte-order mark and a companion
  page of labels, rubrics and sources. `import-reviews` writes only accepted and edited rows into a
  new version 2 set as reviewed ratings: it is atomic and idempotent, re-checks the snapshot guard,
  never changes an existing reviewed rating without `--replace`, and validates the result.
- `examples/reference_rating`: the VEE pilot (a seeded sample over case ids; packets that withhold
  VEE's commentary and every source dated after the snapshot). Design:
  [reference rating](docs/design/reference-rating.md).

### Calibration on synthetic data (deep dive 5)

- `src/mcda_studio/calibration/`: an offline, domain-neutral module. It fits a censoring-aware
  constant-hazard outcome model to a version 2 reference set, with a nested Dirichlet prior centred
  on a profile's stated weights (κ moment-matched to the DD3 conviction bands: 555 / 136 / 47 on
  case 2's top level). It uses its own NumPy sampler (adaptive random-walk Metropolis within Gibbs,
  four chains, PCG64), rank-normalised split-R-hat and bulk ESS, and a fixed convergence gate
  (R-hat < 1.01, ESS ≥ 400).
- Only reviewed ratings count. `--include-proposed` is diagnostic only and says so. Missing is never
  zero, and unknown follow-up is excluded under a stated missing-at-random assumption.
- `mcda calibrate fit` writes a new opt-in sidecar, `weight-samples` version 1, and only for a
  converged run. `mcda calibrate synthetic-check` plants weights and checks recovery. Neither the
  API nor the interface imports the module.
- Validated on synthetic data only: recovery improves with the number of cases, interval coverage,
  simulation-based calibration, and a censoring-bias regression. Nothing is claimed about real
  outcomes. Design: [calibration](docs/design/calibration.md). Proposal (not implemented):
  [sampled weights in SMAA-2](docs/design/sampled-weights-in-smaa.md).
- `examples/vee_bridge/convert.py` now writes its output as UTF-8, so it no longer fails on Windows
  with a non-UTF-8 code page.

### Reference sets version 2 (proposed; needs the owner's approval)

- A second `reference-cases` format with **observation-aware outcomes**: named outcome definitions
  with a window, and per case `event`, `competing_event`, `no_event`, `censored` or `unknown`, so
  what was not watched is never read as what did not happen. Partial dates (`YYYY`, `YYYY-Qn`,
  `YYYY-MM`) for sources and events; the snapshot guard uses the latest day a date can mean.
  Every rating says whether it is `proposed` or `reviewed`.
- `mcda reference-cases outcomes`: counts by status, the share among fully observed cases with a
  Clopper–Pearson interval, the range over every case, and rating coverage per labelled leaf.
  `mcda reference-cases validate` reads either version. Version 1 is unchanged.
- `examples/vee_bridge`: converts a Venture Evidence Explorer cohort audit into a version 2 set,
  mapping only exact rubric equivalences and marking every rating proposed. Design:
  [reference cases, version 2](docs/design/reference-cases.md#version-2-observation-aware-outcomes-proposed).

### Structured judgement (schema 7)

- A weight profile may carry an **elicitation record**: who gave the judgement and when, a
  conviction level, an order per sibling group, optional AHP answers and one reason per weight. It
  stores judgements only; the swing ratings stay the judgement of record.
- **Checks** (read-only): swing, rank-order (ROC) and AHP weights on one footing, orders the
  ratings contradict, AHP consistency ratio and most inconsistent triad, the largest swing-AHP gap,
  missing reasons, and a suggested order. `POST /api/elicitation/check`,
  `mcda elicitation-check`.
- **Proposals**: *conviction* (each nonzero rating within ±15/30/50%) and *ranking only*, each a
  complete SMAA-2 section applied by a person; `POST /api/elicitation/preview` gives decision
  confidence under each before anything is applied.
- **SMAA-2**: a `ranking` weight kind (uniform on the ordered region; mean = ROC weights) and
  `held` children on any group (kept at the profile's weight). Earlier outputs are byte-identical
  (969 of 969).
- **App**: a Judgement tab in Build; the Uncertainty tab states rankings and held children.
- Schema version 7, written only when one of these is used; version-6 files cannot hold them.
  `mcda uncertainty group --rank … --hold …`. Design:
  [structured judgement](docs/design/structured-judgement.md).

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
