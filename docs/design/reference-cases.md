# Reference cases

**Accepted by the owner through the reference cases prompt, 12 November 2026**
(`docs/development/reference-cases-2026-11-12-prompt.md`). The prompt approves exactly one new
stored format, the `reference-cases` sidecar version 1, as specified below. It is opt-in and
read-only: it changes no existing schema, mathematics, fixture or pinned output, and no score,
rank, class or stored project byte.

**Owner amendment, 14 November 2026 (chat):** add an opt-in **leave-one-out** option,
`exclude_same_id` (default false). When true, a case whose id equals the alternative's id is left
out of that alternative's comparison before anything is computed, so it is neither ranked nor
listed as not comparable, and its id is reported in that alternative's `excluded_case_ids`. The
result echoes `exclude_same_id`. Similarities, order and grouping of every other case are
unchanged; with the option off, every result has the same values, order and groups as before
(the response gains the echoed `exclude_same_id: false` and an empty `excluded_case_ids` per
alternative). It exists for sets whose cases are the project's own alternatives, such as the
first-round retrospective, where a company otherwise matches itself exactly.

## Purpose

For an alternative in the open project, show the most similar **reference cases**: past
alternatives rated on the same labelled criteria as of a dated **snapshot**. The analysis gives the
criteria that matched, the ones that differed and, only on request, each case's recorded
**outcome**. It predicts nothing. Similarity is a description of how alike two sets of ratings
are, not a forecast.

## Wording

Engine code and engine-generated text use only *reference case*, *snapshot*, *outcome*,
*context*, *similarity*, *criterion* and *label*. Domain words (stage names, funding words, success
or failure) appear only in example data and the example's README. A unit test checks the fixed
label and limitations for them.

## The `reference-cases` sidecar, version 1

A separate local JSON file. It reuses the evidence models where they fit and adds only what they
cannot express.

- **Envelope:** `kind: "reference-cases"`, `version: 1`, `id`, `title`, `description`,
  `criteria_from` (the id of the project whose criteria tree the labels use), `cases` and
  `sources`. The strict sidecar conventions of the External Proposals format apply: unknown fields
  are refused, versions must be explicit integers, and any version other than 1 is refused with a
  message naming the supported version. `sources` is a map from a source key to a source, as in
  the proposals intake.
- **Source:** the evidence `Source` (`name`, `reference`, `location`, `date`, `description`,
  `classification`) except that `date` may be absent. The date is the day the source was published
  or, for an archived capture, captured. Dates are explicit `YYYY-MM-DD` strings; numbers and other
  formats are refused.
- **Case:** `id`, `name`, `snapshot_date` (required), `context` (an ordered list of
  `{label, value}` strings shown exactly as written), `claims`, `ratings` and `outcome` (required;
  "outcome not found" is itself a label an author can record).
- **Claim:** `id`, `text` (the author's own words), `source` (a source key) and `origin`, the
  evidence `Origin` (`kind` human or tool, `proposed_by`, optional tool identity). The cell-specific
  fields of the evidence `Claim` (alternative, criterion, value, unit, review) do not apply to a
  statement about a past case, so the claim is its own small model; `Source` and `Origin` are
  reused unchanged in meaning. Claims live inside their case, and a rating or outcome may cite only
  its own case's claims.
- **Rating:** `criterion_id`, `label` (exactly one of that leaf's scale labels), `claim_ids` (one
  or more) and an optional `note`. A leaf with no rating is **missing, not zero**.
- **Outcome:** `label` (free text from the data), `as_of`, `note` and `claim_ids` (may be empty).

### Validation (refused with a precise path)

| Code | Path | Rule |
| --- | --- | --- |
| `REFERENCE_PROJECT_MISMATCH` | `criteria_from` | must equal the project's id |
| `REFERENCE_DUPLICATE_CASE` | `cases[i].id` | case ids are unique |
| `REFERENCE_DUPLICATE_CLAIM` | `cases[i].claims[j].id` | claim ids are unique in a case; the first stands |
| `REFERENCE_UNKNOWN_SOURCE` | `cases[i].claims[j].source` | the source key exists |
| `REFERENCE_UNKNOWN_CRITERION` | `cases[i].ratings[j].criterion_id` | a leaf criterion of the project |
| `REFERENCE_NUMERIC_CRITERION` | same | version 1 rates labelled criteria only, and says so |
| `REFERENCE_UNKNOWN_LABEL` | `cases[i].ratings[j].label` | one of that leaf's labels, exactly |
| `REFERENCE_DUPLICATE_RATING` | `cases[i].ratings[j].criterion_id` | one rating per leaf per case |
| `REFERENCE_UNKNOWN_CLAIM` | `...claim_ids[k]` | the cited claim exists in this case |
| `REFERENCE_OUTCOME_BEFORE_SNAPSHOT` | `cases[i].outcome.as_of` | on or after the snapshot date |
| `REFERENCE_UNDATED_SOURCE` | `cases[i].ratings[j].claim_ids[k]` | the snapshot guard, below |
| `REFERENCE_SOURCE_AFTER_SNAPSHOT` | same | the snapshot guard, below |

**The snapshot guard (the hindsight rule).** Every claim cited by a *rating* must rest on a source
dated **on or before** that case's snapshot date. An undated source cannot support a rating; a
dated archived capture can. Outcome claims may be dated after the snapshot, and may be undated.
The guard is enforced in the service, so no analysis runs on a set that breaks it. It checks dates,
not whether a claim was knowable at the time.

## The computation (Python only)

Inputs: the project, the analysis configuration (for the active weight profile), the reference
set, an optional `alternative_id` (default every alternative), `k` (default 3, 1 to 20),
`weighting` (`"equal"`, the default, or `"profile"`), `min_shared_fraction` (default `"1/2"`, a
fraction or decimal string greater than 0 and at most 1), `include_outcomes` (default false) and
`exclude_same_id` (default false; amendment of 14 November 2026).

- **Compared criteria L:** leaves with an ordinal or categorical scale whose global weight in the
  active profile is positive, from the engine's own `compute_weights` (never re-implemented).
  Zero-weight leaves and numeric leaves are not compared and are not listed.
- **Per-leaf gap**, with `fractions.Fraction` throughout: ordinal `|i − j| / (n − 1)` over the label
  positions and the scale length (a one-label scale has gap 0); categorical 0 if equal, else 1.
- **Shared leaves S:** leaves in L rated on both sides. If S is empty, or `|S| / |L|` is below the
  threshold, the case is **not comparable**: it is listed with both counts and its fraction, and
  never ranked. Exactly at the threshold is comparable.
- **Distance:** `"equal"` is the arithmetic mean of the gaps over S; `"profile"` weights each gap
  by the leaf's global weight (the float converted exactly to a fraction), renormalised over S.
  Similarity is `1 − distance`.
- **Order:** similarity descending, then `|S|` descending, then case id ascending, comparing exact
  fractions. The top `k` are returned with the count of the remaining comparable cases.
- **Each returned case** gives its context, similarity, shared and compared counts and the leaves
  of L in four groups: **same** label; **one step** apart (an ordinal gap of exactly one
  position); **further** apart (more than one step, or a categorical mismatch); and **not
  compared**, each saying whether the alternative, the reference case or both lack a rating. Every
  row gives both labels, the steps apart on an ordered scale (none on a categorical scale or where
  a side is missing), the exact gap and the case's rating note.
- **Numbers** leave the engine as in decision impact: a float (`similarity`), plus the exact value
  as a `"p/q"` string (`similarity_exact`, `gap`, `shared_fraction`, `min_shared_fraction`).
- **Outcomes** are in the result only when `include_outcomes` is true.
- **The fixed label**, in every result: similarity is measured only on the rated criteria as each
  case stood at its snapshot; it says nothing about how the alternative will turn out; outcomes are
  context chosen by the reference set's author; a small set is not representative and public
  sources under-report cases that closed.
- **Limitations**, in every result: numeric criteria are not compared in version 1; every compared
  criterion counts equally unless profile weighting is chosen; the snapshot guard checks dates, not
  knowability; the set's author knew the outcomes when rating.

Refusals: an invalid reference set raises a validation failure with every issue (HTTP 422,
`VALIDATION_FAILED`, each issue's path in the details); an unknown alternative is
`UNKNOWN_REFERENCE`; `k`, `weighting` and the threshold out of range are refused. A configuration
whose weight profile cannot be resolved or weighted gives `status: "unavailable"` with the reason,
as decision impact does. An alternative's own label that is not on its scale is a model error. The
project's measurements need not be complete: missing cells are simply not shared.

## Surfaces

- **CLI:** `mcda reference-cases PROJECT --reference-cases FILE [--alternative ID] [--k N]
  [--weighting equal|profile] [--min-shared-fraction F] [--include-outcomes] [--exclude-same-id]
  [--profile ID] [--scenario ID]`, printing JSON (the same command is also `mcda reference-cases run`), and
  `mcda reference-cases validate FILE --project PROJECT`, printing the validation JSON and exiting 1
  when the set is invalid.
- **API:** `POST /api/reference-cases` and `POST /api/reference-cases/validate`, tag `analysis`,
  stateless: the request carries the project and the sidecar.
- **Interface:** a Reference cases panel in the Analyst view (a later development in the same run).
  Since 14 November 2026 it also has **Leave out this alternative's own case**, off by default,
  which re-requests with `exclude_same_id` and states when the engine left a case out.

## Excluded from version 1

Numeric leaves; combining several reference sets; storing a reference set inside a project; any
use of outcomes in a score, weight, rank or class; any learning from outcomes; editing a reference
set in the application.

## Verification

- Unit tests for every validation error and its path, the snapshot guard (one day after refused,
  the same day accepted, undated refused for a rating and accepted for an outcome), ordinal gaps at
  both ends of four- and five-label scales, a categorical mismatch, missing on either side, the
  threshold exactly at one half and just under, a hand-computed example under both weightings, the
  tie-break order including an exact tie that floats reverse, `k` bounds, outcomes absent by
  default, zero-weight leaves, and unchanged project bytes, fingerprint and ETag.
- An independent oracle (`tests/regression/reference_cases_oracle.py`) that never imports the
  application derives the weights from the raw swing ratings and recomputes every similarity,
  order, group and not-comparable case; the service agrees exactly on hand-built and generated
  sets.
- API and CLI tests like decision impact's.
- A new mutant group, `reference_cases`, in `scripts/mutation_check.py`.

## Version 2: observation-aware outcomes (proposed)

**Proposed on 25 November 2026 (deep dive 4 of the evidence-backed weighting plan); awaiting the
owner's approval under rule 3 before it merges.** Version 2 is a second stored format next to
version 1. It changes nothing in version 1: version 1 files are read, validated and compared exactly
as before, with the same messages, and the similarity analysis reads version 1 only.

### Why

An outcome model (deep dive 5) must keep three things apart that version 1's free-text outcome
cannot: the event happened, the case was watched long enough to know it did not, and the case was
not watched long enough to know. Treating an unwatched case as a failure biases every rate down, and
dropping it biases toward cases that were followed. Public dates are also often a month or a year,
and a tool-proposed rating must not be mistaken for a reviewed one.

### What version 2 adds

- **Envelope:** `version: 2`, `observed_through` (the day the follow-up was compiled) and
  `outcome_definitions`, a map from a key to `{label, window_months (1 to 240), description}`. At
  least one definition.
- **Partial dates** for source dates and event dates: `YYYY`, `YYYY-Qn`, `YYYY-MM` or `YYYY-MM-DD`.
  The snapshot date and `observed_through` stay exact days.
- **Rating:** version 1's rating plus a required `review`, `proposed` or `reviewed`.
- **Case:** version 1's case with `outcomes` (a list of observations, at most one per definition)
  in place of `outcome`.
- **Observation:** `definition`, `status`, `event_date`, `observed_through`, `claim_ids`, `note`.

| Status | Meaning | Required | Refused |
| --- | --- | --- | --- |
| `event` | the event happened inside the window | `event_date`, a claim | `observed_through` |
| `competing_event` | another terminal event ended the case inside the window first | `event_date`, a claim | `observed_through` |
| `no_event` | watched to the end of the window, no event seen | `observed_through` | `event_date` |
| `censored` | watched to a day before the window ended, no event seen | `observed_through` | `event_date` |
| `unknown` | no follow-up | | `event_date`, `observed_through` |

The window opens on the snapshot day and ends `window_months` calendar months later (a day past the
end of the target month moves to its last day).

### Validation, in addition to every version 1 rating rule

| Code | Rule |
| --- | --- |
| `OUTCOME_UNKNOWN_DEFINITION`, `OUTCOME_DUPLICATE_DEFINITION` | a known definition, once per case |
| `OUTCOME_EVENT_DATE_REQUIRED`, `OUTCOME_EVENT_UNSUPPORTED` | an event or competing event has a date and a claim |
| `OUTCOME_UNEXPECTED_EVENT_DATE`, `OUTCOME_UNEXPECTED_OBSERVED_THROUGH`, `OUTCOME_OBSERVED_THROUGH_REQUIRED` | the table above |
| `OUTCOME_EVENT_BEFORE_SNAPSHOT` | the event date cannot be after the snapshot |
| `OUTCOME_EVENT_AFTER_WINDOW` | the event date is certainly after the window (record `no_event`) |
| `OUTCOME_EVENT_DATE_TOO_COARSE` | the partial date could fall outside the window (record a finer date or `censored`) |
| `OUTCOME_OBSERVED_BEFORE_SNAPSHOT` | watched to before the snapshot (record `unknown`) |
| `OUTCOME_WINDOW_NOT_COMPLETE` | `no_event` watched to before the window end (record `censored`) |
| `OUTCOME_CENSORED_AFTER_WINDOW` | `censored` watched through the window end (record `no_event`) |
| `OUTCOME_AFTER_FOLLOW_UP`, `REFERENCE_SNAPSHOT_AFTER_FOLLOW_UP`, `REFERENCE_SOURCE_AFTER_FOLLOW_UP` | nothing is seen after `observed_through` |

**The snapshot guard on partial dates:** a source supports a rating only if the *last* day its date
can mean is on or before the snapshot. `2021-10` cannot support a rating at a 27 October 2021
snapshot; `2021-09` can.

### The summary (bookkeeping, not a model)

`mcda reference-cases outcomes FILE --project PROJECT` validates the set, then gives, per
definition: the count of each status and of cases with nothing recorded; the share of events among
fully observed cases (event, no event, competing event), as an exact fraction with a 95%
Clopper–Pearson interval from the engine's own `clopper_pearson`; and the **range** the share over
every case could take, from every unresolved case ending without the event (lowest) to every one
ending with it (highest). It also gives **coverage**: for every labelled leaf, how many cases rate it,
reviewed and proposed, and how many do not. A fixed label says the counts are not a chance of success
for any alternative and that censored and unknown cases are never counted as no event; the
limitations name the follow-up-time bias a time-to-event model would have to correct.

`mcda reference-cases validate` reads either version by its `version` field.

### Not in version 2

The similarity analysis on version 2 sets; numeric leaves; storing a set inside a project; any use of
outcomes in a score, weight, rank or class. Learning from outcomes is deep dive 5, behind its own
synthetic validation and gate.

### Verification

`tests/unit/test_reference_outcomes.py`: partial dates and calendar months against a walk over
every day of the calendar, every outcome code, inclusive window boundaries, the snapshot guard on
each partial form, version 1 messages unchanged, the summary against hand counts with SciPy's beta
quantiles as the interval oracle, coverage by review state, read-only behaviour, and the commands.
The mutant group `reference_outcomes` (15 mutants) in `scripts/mutation_check.py`. The VEE bridge
in `examples/vee_bridge` has its own test on a synthetic audit.
