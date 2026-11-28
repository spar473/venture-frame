# Sampled weights in SMAA-2

**Status:**
- Proposed on 25 November 2026 (deep dive 5, phase 4).
- **Approved by the owner in the DD7 session prompt of 26 November 2026**, which also approved
  schema 8 for saved `"samples"` configurations and a `basis` field on `weight-samples` v1.
- **Implemented** on `feature/conviction-prior`. The conviction prior came first (decision 4):
  `mcda calibrate prior`.
- Where the code had to settle what this note left open, see
  [Implementation notes](#implementation-notes) at the end. Every existing output is
  byte-identical: 969 of 969 against `feature/calibration`.

## Why

DD3 promised a true `Dirichlet(κ·w)` conviction. It deferred that because the engine has no
portable gamma quantile, and SMAA-2 draws every weight from uniforms through quantile functions.
DD5 now produces weight vectors as data: the `weight-samples` sidecar
([calibration](calibration.md)). This proposal lets SMAA-2 read those vectors instead of drawing
weights.

The sidecar can hold any of three things:

- a calibrated posterior;
- a pure prior, from a run with no entering case;
- in future, samples written by a dedicated "conviction prior" command.

So the same path serves DD3's conviction and DD5's calibration. The sampling happens in
`calibration`, which may use NumPy's gamma draws because it does not promise cross-platform bits.

## Proposal

### 1. A new weight source, chosen per run

Two new SMAA-2 method parameters go in `analysis.parameters`, which the method owns:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `weight_source` | `"stated"` (today's behaviour: the `smaa` section's weight groups) or `"samples"` | `"stated"` |
| `weight_samples_sha256` | the SHA-256 of the sidecar file the run must use; required with `"samples"` | none |
| `resample` | allow more SMAA draws than stored vectors (below) | `false` |

- **The default changes nothing.** `"stated"` is the default and is never written when it is the
  default. Configurations that do not name these keys keep their bytes.
- **Where the file comes from.** The sidecar is never stored in the project. It travels next to it:
  - CLI: `mcda evaluate --method smaa2 --weight-samples FILE`;
  - API: the stateless `POST /api/evaluate` request gains an optional `weight_samples` body field,
    as reference sets already travel.
  - The digest parameter pins which file a saved configuration means. A file with another digest is
    refused (`WEIGHT_SAMPLES_DIGEST_MISMATCH`).

### 2. What the file must satisfy (refused otherwise, with a code)

- `kind` must be `weight-samples` and `version` must be 1, through the sidecar model, which is
  already strict.
- `usable` must be true, so diagnostic and insufficient runs are refused
  (`WEIGHT_SAMPLES_NOT_USABLE`). A converged run is already guaranteed, because the model refuses
  `converged: false`.
- `leaf_ids` must equal the project's leaves in tree order (`WEIGHT_SAMPLES_TREE_MISMATCH`).
- `weight_profile_id` must exist in the project.
- **Model identity.** The sidecar's `project_fingerprint` covers the whole project, including
  `analysis` and `smaa`. So choosing the new parameters would itself change it. The proposal
  compares a *weights-basis digest* instead: the project fingerprint computed with `analysis`,
  `smaa`, scenarios and decision records set aside. It is derived, never stored.
  - A mismatch is refused (`WEIGHT_SAMPLES_STALE`).
  - Because version 1 stores only the full fingerprint, a version 1 file is accepted only when that
    full fingerprint matches the project as it was fitted.
  - Recommended follow-up: `weight-samples` version 2 adds `weights_basis_sha256`. That would be a
    new format, and so another rule-3 decision.
- **The run's `uncertainty` setting.** With `"samples"`, a `performance` run ignores the file, as it
  ignores stated weight groups today. A `preference` or `combined` run uses the file *instead of*
  every stated weight group, and the basis sentence and the record say so. Stating weight groups
  and using samples in one run is refused (`WEIGHT_SOURCE_CONFLICT`), so the two measures over
  weights are never mixed silently. The refusal names the fix: save a copy without the weight
  groups (`mcda uncertainty group <project> --fixed -o <copy>`, plus `--parent <id>` for each
  group below the top level) and draw the prior from that copy, so the sidecar's project
  fingerprint is the copy's. `mcda calibrate prior` still writes a sidecar for a project with
  weight groups (exit 0, the same stdout), and warns on stderr with the same command.

### 3. Matching draws to the sample count (never recycled silently)

Let `S` be SMAA-2's `samples` and `N` the sidecar's stored vectors (4 000 by default).

- **`S ≤ N`:** draw `S` distinct indices without replacement from `{0, …, N − 1}`. The draw is a
  partial Fisher–Yates shuffle on a separate stream, `PCG64(seed).jumped(2)`.
  - The main stream `PCG64(seed)` is untouched, so measurement-cell draws are the stream's first
    columns in this mode.
  - Index draws come only from integers, so they are the same on every platform.
- **`S > N`:** refused with `WEIGHT_SAMPLES_TOO_FEW` unless `resample: true`. With it, the draw is
  `S` indices with replacement from the same stream, and every output carries this sentence:

  > "S draws reuse N stored weight vectors; interval widths for weight-driven frequencies reflect
  > S draws but the weights carry the information of N."

  - With `resample`, the Clopper–Pearson intervals of a preference run are computed with
    `n = min(S, N)`, so reuse cannot make them look narrower. A combined run keeps `n = S`, and the
    sentence states the caveat.
- **Record.** The record gains the sidecar digest, `N`, the index method (`without replacement` or
  `with replacement`) and the stream name `smaa2.pcg64.v1+samples`.

### 4. Byte identity

Every existing output must stay byte-identical: the 969-output manifest of `scripts/byte_identity.py`,
compared against the commit before the change (and with `--portable` across machines, as today).

- No existing configuration names `weight_source`, and the stated path's code is not touched.
- The new branch sits behind `if parameters.weight_source == "samples"` before `_group_locals`.

### 5. Schema and configuration

- **Stored project schema: unchanged (still 7).** The new keys are method parameters.
  - `SmaaParameters` uses `extra="ignore"`, and validation reports unknown keys as
    `UNKNOWN_PARAMETER` warnings. So an older reader would warn about, and ignore, `weight_source`,
    and silently run on stated weights.
  - To prevent that, a project that saves `weight_source: "samples"` is written at the lowest
    version that can hold it, which this proposal sets to **8**. A version-7 reader then refuses the
    file instead of running a different analysis.
  - This is a stored-schema decision for the owner. The alternative is to keep version 7 and accept
    the warn-and-ignore behaviour in older readers.
- `docs/api/openapi.json` gains the optional `weight_samples` request field and the new parameter
  names. The generated frontend types follow (`npm run gen:types`).
- The interface shows the source and the record sentence on the Acceptability panel. The browser
  computes nothing.

## Tests

- **Oracle, Dirichlet path agreement.**
  - On the analytic two-option model (the 3/7 case) and on case 2's top level: draw `N = 100 000`
    vectors from `Dir(1, …, 1)` (the `unknown` kind's measure) with an independent NumPy sampler.
    Write them as a sidecar, run SMAA-2 with `weight_source: "samples"` and compare every rank and
    class frequency with the existing `unknown` path at the same `S`.
  - Each frequency must lie inside the other's exact 99.9% Clopper–Pearson interval, and the 3/7
    case must contain 4/7 and 3/7.
  - The same with `Dir(κ·w0)` for κ = 47, 136 and 555, compared against an independent NumPy replica
    of SMAA-2 that draws Dirichlet weights directly. This is the DD3 promise, checked.
- **Byte identity:** the 969 outputs, unchanged.
- **Matching:**
  - `S ≤ N` uses distinct indices (exact);
  - `S > N` is refused without `resample`;
  - with `resample` the record and sentence appear, and intervals use `min(S, N)` in a preference
    run;
  - the main stream is unchanged by the index stream (cell draws equal a run with no weight
    uncertainty at the same seed).
- **Refusals:** every new code, including a diagnostic sidecar, a mismatched tree, a stale model, a
  wrong digest and a stated-groups conflict.
- **Reproducibility:** the same seed and file give identical bytes; another seed differs.
- **Read-only:** the project bytes and fingerprint do not change when a run uses samples.

## Mutants (proposed group `smaa_samples`)

1. Indices drawn with replacement when `S ≤ N`.
2. `S > N` accepted without `resample`.
3. Resampled intervals use `n = S` in a preference run.
4. Index draws taken from the main stream, which shifts cell draws.
5. A diagnostic (`usable: false`) sidecar accepted.
6. The digest check skipped.
7. The tree check skipped.
8. Stated weight groups and samples mixed in one run.
9. A performance run reads the samples.
10. The record omits the index method.
11. The `"stated"` default written into configurations, which breaks byte identity.
12. Held children of a stated group applied on top of sampled weights.

## Decisions for the owner

1. Approve the weight source as specified, or ask for changes.
2. Schema 8 for saved `"samples"` configurations, or keep 7 with warn-and-ignore in older readers.
3. Whether to plan `weight-samples` version 2 with a weights-basis digest.
4. Whether a "conviction prior" command (samples from `Dir(κ·w0)` with no reference set) should come
   first. It would deliver DD3's promise before any outcome data exist.

## Implementation notes

What the code settles that the proposal left open:

- **The conviction prior.** `mcda calibrate prior PROJECT --profile ID [--conviction | --kappa]
  [--draws N] [--seed S] --out FILE` writes a sidecar with `basis: "prior"`.
  - It draws exact `Dirichlet(κ·w0)` per varying sibling group (`methods/nested_dirichlet.py`).
  - κ is moment-matched to the conviction (555 / 136 / 47 on case 2's top level), or stated.
  - Held and zero-weight children behave as DD3 says.
  - `--draws` defaults to 10,000, SMAA-2's default sample count, so S ≤ N by default.
- **`basis`.** `weight-samples` v1 gains `basis`, `"prior"` or `"calibrated"`.
  - The reference-set fields and the outcome model's priors and draws are required only for a
    calibrated basis, and refused for a prior basis.
  - A file without `basis` reads as calibrated.
- **Digest.** `weight_samples_sha256` is the SHA-256 of the sidecar's canonical JSON (sorted keys,
  UTF-8, `(',', ':')`), not of the file's bytes. A file and the same sidecar sent in an API
  request therefore pin the same digest.
- **Staleness without a weights-basis digest.** v1 stores only the full project fingerprint. The
  check compares it with the project's fingerprint after removing the three source parameters,
  which reconstructs the project as it was when the samples were drawn.
  - Workflow: save the method and SMAA-2 settings first, then draw, then save `weight_source` and
    the digest.
  - Any other change makes the file stale (`WEIGHT_SAMPLES_STALE`).
  - The weights-basis digest of decision 3 is not added; it was not approved.
- **Codes the proposal described but did not name:**
  - `WEIGHT_SAMPLES_DIGEST_REQUIRED`: samples without a digest;
  - `WEIGHT_SAMPLES_MISSING`: no sidecar supplied;
  - `WEIGHT_SAMPLES_UNKNOWN_PROFILE`;
  - `WEIGHT_SAMPLES_PROFILE_MISMATCH`: the run's profile is not the sidecar's. This one is a
    conservative addition.
- **Resampled intervals.** In a resampled preference run each count is scaled to N before its
  Clopper–Pearson interval is computed.
- **The agreement oracle.** It compares two sampled estimates with a two-proportion test at a
  family-wise α of 0.001. Requiring each estimate inside the other's 99.9% interval would fail
  about 2% of comparisons for correct code. Against the NumPy replica (ten times the draws), the
  engine's interval at 1 − 0.001/m must contain the replica's frequency.
- **The application.** The Judgement tab's decision-confidence comparison has a "Weight
  uncertainty" choice: conviction bands, ranking only, or true conviction (Dirichlet κ).
  - The true-conviction column comes from the elicitation preview, which draws the prior and runs
    SMAA-2 through the samples source server-side. It shows the κ used.
  - It cannot be "used as the weight uncertainty" from the app, because the sidecar lives in no
    project. Saving a samples configuration goes through the CLI (`mcda evaluate
    --weight-samples`, or saved parameters plus the file).
- **Tests and mutants.**
  - `tests/unit/test_smaa_samples.py` and `tests/unit/test_conviction_prior.py`;
  - `tests/regression/test_smaa_samples_reference.py`;
  - the `smaa_samples` mutant group (18 mutants).
