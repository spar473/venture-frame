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
`rank_high`, `band_share`, `rank1_share`, and the decision-confidence fields `first`, `top_ranks`
and `classes` below), `top_k`, `confidence`, `groups`, `basis`, `label` and `confidence_label`. The
label: "Rank ranges show where each alternative's rank falls in most draws under the uncertainty
stated in the project. They are not probabilities of success or predictions."

## Decision confidence

**Status:** accepted by the owner, 25 November 2026 (deep dive 1 of the evidence-backed weighting
plan), including the regenerated golden report. It changes no mathematics, project, sidecar or
method output, and SMAA-2 itself is unchanged.

Rank ranges answer "which ranks are too close to call". Decision confidence states the same draws
as plain shares, per alternative and per profile, so a partner can read how settled each reading
is:

- **`first`**: the share of draws at rank 1, tied first places included (the same share as
  `rank1_share`).
- **`top_ranks`**: the share of draws whose competition rank is at most `top_k`, where `top_k` is
  `min(3, number of alternatives)`. Each draw gives one rank per alternative, so this is the sum
  of the first `top_k` rank counts.
- **`classes`**: one share per class, best class first, from SMAA-2's class acceptabilities;
  empty without classes. Every draw falls in exactly one class, so a row's class shares sum to 1.

Each is a `ShareEstimate` (`share`, `lower`, `upper`). In a sampled run, `lower` and `upper` are
the exact two-sided Clopper–Pearson interval at SMAA-2's `confidence` (0.95 by default): for
`first` and `classes` they are SMAA-2's own intervals, and for `top_ranks` they are computed from
the summed count with the same function. Under exact enumeration the shares are probabilities,
`fsum`med, with no interval, and `confidence` is `None`.

The `confidence_label`: "Decision confidence is the share of draws, under the uncertainty stated
in the project, in which an alternative is first, is in the top ranks or falls in a class. It
describes how settled the screening reading is, not a probability of success."

**What it is about.** The shares describe the uncertainty *stated in the project*, nothing more.
In case 2 that is 87 measurement ranges and top-level swing-rating intervals the study author
typed in, so the shares measure the consequence of those stated ranges. They become a statement
about partners' real uncertainty only when weights are elicited with ranges (deep dive 3).

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
- Decision confidence: a table under the screening summary in the app (First, Top 3, one column
  per class, each share with its interval in brackets), and a "Decision confidence by profile"
  section in the HTML report with the same table per profile. Shares display as whole percents,
  never 0 % or 100 % for a share that is neither ("< 1%", "> 99%"); interval ends are widened
  outward to whole percents (floor and ceiling), so rounding never narrows an interval.

## Case 2 (venture study), default run

| Profile | Groups (MAVT order) | Leader's rank-1 share |
| --- | --- | --- |
| Technical/IP-led | [Upstock] · [Wellumio, Hectre, Wych, Captivate Technology, Scentian Bio, Hnry] · [Kea Aerospace] | 0.9302 |
| Commercial/traction-led | [Upstock] · [Hectre, Wych, Hnry, Wellumio, Captivate Technology, Scentian Bio] · [Kea Aerospace] | 0.9247 |

Technical/IP-led ranges, in MAVT order: 1–1, 2–6, 2–4, 2–4, 4–7, 3–6, 5–7, 8–8.

Decision confidence, Technical/IP-led (shares of 10,000 draws):

| Alternative | First | Top 3 | Prioritise for diligence | Monitor | Not now |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstock | 0.9302 | 0.9991 | 0.9744 | 0.0256 | 0 |
| Wellumio | 0.0060 | 0.2800 | 0.0599 | 0.9401 | 0 |
| Hectre | 0.0061 | 0.6907 | 0.1224 | 0.8776 | 0 |
| Wych | 0.0555 | 0.7700 | 0.3570 | 0.6430 | 0 |
| Captivate Technology | 0.0017 | 0.0910 | 0.0156 | 0.9799 | 0.0045 |
| Scentian Bio | 0.0004 | 0.1138 | 0.0064 | 0.9936 | 0 |
| Hnry | 0.0001 | 0.0554 | 0.0072 | 0.9900 | 0.0028 |
| Kea Aerospace | 0 | 0 | 0 | 0.2810 | 0.7190 |

Wych, fourth by MAVT, is in *Prioritise for diligence* in about a third of draws; Kea Aerospace,
last, is in *Monitor* in more than a quarter.

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
- Decision confidence (`tests/regression/test_decision_confidence_reference.py`): on case 2's real
  output under both profiles, every share is recounted from SMAA-2's rank and class counts, and
  every sampled interval is recomputed with SciPy's beta quantiles (not the engine's), agreeing
  within 10⁻⁹; class shares sum to the draws; classes come best first. Exact enumeration matches
  `smaa_oracle`'s exact probabilities with no interval. Tied first places count as first
  (`tie_case`); a two-alternative project counts its top 2. The report test parses each decision
  confidence cell and rebuilds it from the counts with its own formatter.
- Six more mutants under `--robustness-only`: top ranks one rank short, `top_k` not capped at the
  number of alternatives, sole first in place of tied first, classes not best first, a fixed 90 %
  interval level, and report intervals rounded inward.
- Nine mutants (`scripts/mutation_check.py --robustness-only`): an interval that drops its last
  rank, `>` for `≥` on coverage, touching ranges not overlapping, a sweep in MAVT order, ties that
  ignore the MAVT rank, enumeration without its tolerance, a missing "no stated uncertainty"
  reason, transient run settings used, and a modified project.

## Limits

Rank ranges reflect only the uncertainty stated in the project, under SMAA-2's own assumptions
(independent draws, or shared draws where stated). They are not probabilities of success,
forecasts or a proof that an order holds for every weight. The coverage is fixed at 0.80. Report
generation runs SMAA-2 once per saved profile (about 0.2 s each for case 2).
