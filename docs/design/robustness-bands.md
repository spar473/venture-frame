# Robustness bands: which ranks are too close to separate

**Status:** accepted by the owner through the partner-view prompt, 24 November 2026 (design
decisions D5 and D6); implemented as `screening_robustness_service` and
`POST /api/screening-summary/robustness`. It adds no stored format and changes no mathematics,
project, sidecar or existing output, and SMAA-2 itself is unchanged.

## Purpose

The screening summary ranks alternatives by their MAVT value under one profile. When the project
states how uncertain its inputs are, some of those ranks are too close to call. Robustness bands
say so in one reading: for each alternative, the run of ranks it takes in most draws, and which
alternatives overlap. It is a reading of SMAA-2's rank acceptabilities, not a new method.

## Definitions

- **Run.** SMAA-2 runs unchanged through `analysis_service.evaluate_project`, with the project's
  `smaa` section, the selected profile and scenario, and the saved SMAA-2 parameters when the
  saved method is SMAA-2, otherwise SMAA-2's documented defaults (`uncertainty: "combined"`,
  10 000 draws, seed 0). The view's transient run settings are never used.
- **Rank range.** For an alternative, the shortest run of consecutive ranks `[low, high]` whose
  summed rank acceptability is at least the coverage, **0.80**. Among equally short runs, the one
  containing the alternative's MAVT rank wins, then the lowest ranks. `band_share` is the share
  of draws the range actually holds.
- **Arithmetic.** Sampled runs compare integer counts exactly: `held × 5 ≥ 4 × draws`, so there is
  no floating-point edge at exactly 80 %. Exact enumeration reports float probabilities, which are
  summed with `math.fsum` and accepted within 10⁻¹² of 0.80 (0.1 + 0.7 is 0.7999999999999999 in
  binary).
- **Groups.** Connected components of the interval graph: two ranges overlap when they share at
  least one rank (`[2–4]` and `[4–7]` overlap; `[2–3]` and `[4–5]` do not). Components are found
  by sweeping the ranges in order of `low`, never in MAVT order, which could join ranges that do not
  overlap. Members and groups are listed in MAVT rank order. A group of one means the alternative
  is separated; a group of several is "too close to separate".
- **`rank1_share`** is the rank-1 acceptability, tied first places included. `mavt_rank` is
  SMAA-2's reference ranking, MAVT at the entered values and the profile's ratings, which equals the
  screening summary's rank.
- **Unavailable.** A project with no `smaa` section returns `available: false` with the reason "No
  stated uncertainty in this project." A run SMAA-2 refuses (for example `NOTHING_UNCERTAIN`, a
  missing measurement or the work limit) returns `available: false` with the engine's message and
  issue codes. Uncertainty is never invented.

## What varies, and the basis sentence

Every result carries `basis`, one plain sentence built from the run: what varied, what was fixed,
the number of draws and seed, and whether the parameters were saved or defaults. A group stated as
`unknown` or `rating_intervals` **replaces the selected profile's ratings for that group** in a
`combined` or `preference` run, and the sentence says so. For the venture study (case 2) the
top-level rating intervals cover both partners' top-level ratings, so under the default run the
top-level weights are the same draws for both profiles and only the weights below the top level
are the selected profile's:

> Varied: 87 stated measurement ranges; the swing ratings at the top level within stated
> intervals, in place of this profile's. Fixed: other weights (this profile's), value functions,
> class boundaries. 10,000 draws, seed 0, SMAA-2 defaults (no saved SMAA-2 parameters).

## Response

`ScreeningRobustness`: `project_fingerprint`, `configuration` (as requested), the resolved
`weight_profile_id` and `scenario_id`, `available`, `reason`, `issue_codes`, `coverage`, `run`,
`evaluation`, `draws` (none under exact enumeration), `seed`, `parameters` (`saved` or
`defaults`), `rows` (in MAVT rank order: `alternative_id`, `name`, `mavt_rank`, `rank_low`,
`rank_high`, `band_share`, `rank1_share`), `groups`, `basis` and `label`. The label: "Rank ranges
show where each alternative's rank falls in most draws under the uncertainty stated in the project.
They are not probabilities of success or predictions."

## Surfaces

- `POST /api/screening-summary/robustness` (draft project) and
  `POST /api/projects/{project_id}/screening-summary/robustness` (stored project). Both are POST
  routes behind `CrossOriginGuard`, with no exemption: a same-origin browser request is served and
  a cross-site one is refused with 403 `CROSS_ORIGIN_REFUSED`.
- The screening summary (Analyst and Builder views): the Rank column reads `2 (2–6)` once bands
  arrive, rows in a multi-member group share a marker labelled "Too close to separate", and a lead
  sentence above the table is built from these fields only. Bands load after the summary renders,
  through the keyed, abortable request pattern, and are hidden during a what-if trial.
- The HTML screening report: the rank range in each profile's table and one line per group.

## Case 2 (venture study), default run

| Profile | Groups (MAVT order) | Leader's rank-1 share |
| --- | --- | --- |
| Technical/IP-led | [Upstock] · [Wellumio, Hectre, Wych, Captivate Technology, Scentian Bio, Hnry] · [Kea Aerospace] | 0.9302 |
| Commercial/traction-led | [Upstock] · [Hectre, Wych, Hnry, Wellumio, Captivate Technology, Scentian Bio] · [Kea Aerospace] | 0.9247 |

Technical/IP-led ranges, in MAVT order: 1–1, 2–6, 2–4, 2–4, 4–7, 3–6, 5–7, 8–8.

## Verification

- An independent oracle (`tests/regression/robustness_oracle.py`, no `mcda_studio` import) lists
  every interval and chooses by one sort key, and forms groups with a union-find over every
  overlapping pair. The service matches it exactly on hand-built acceptabilities (one-hot,
  uniform, ties, the exact 80 % boundary, binary rounding under enumeration, the MAVT-order sweep
  trap), on 300 generated matrices, and on case 2's real SMAA-2 output under both profiles.
- Properties: `low ≤ high`, `band_share ≥ 0.80`, no shorter run reaches the coverage, groups
  partition the alternatives, the same seed gives the same result, looking never changes the
  project or the screening summary.
- API tests for both routes, the unavailable state, and the cross-site guard.
- Nine mutants (`scripts/mutation_check.py --robustness-only`): an interval that drops its last
  rank, `>` for `≥` on coverage, touching ranges not overlapping, a sweep in MAVT order, ties that
  ignore the MAVT rank, enumeration without its tolerance, a missing "no stated uncertainty"
  reason, transient run settings used, and a modified project.

## Limits

Rank ranges reflect only the uncertainty stated in the project, under SMAA-2's own assumptions
(independent draws, or shared draws where stated). They are not probabilities of success,
forecasts or a proof that an order holds for every weight. The coverage is fixed at 0.80. Report
generation runs SMAA-2 once per saved profile (about 0.2 s each for case 2).
