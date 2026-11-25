# Reference rating: reviewed ratings for reference cases (deep dive 6)

**Status:** set up in the DD6 session prompt of 26 November 2026. Pre-approved scope:
- venture-specific tooling under `examples/reference_rating/`;
- a new opt-in sidecar format, `rating-proposals` version 1, because nothing existing fits (below);
- tests and docs.

It changes no method maths, no stored project schema, neither `reference-cases` format and not
`weight-samples`. The application never calls a model: proposals come from outside (in the pilot,
from Claude Code during a session), and nothing becomes a reviewed rating until a person accepts it
in the review sheet.

## Why

Calibration ([calibration](calibration.md)) needs reference cases rated on the project's own
criteria, as they stood at their snapshot date, by ratings a person has reviewed. The VEE cohort
does not have that today:
- it rates 2 of the screening study's 17 labelled leaves, both zero-weight mandate gates;
- none of its ratings is reviewed;
- so `mcda calibrate fit` refuses.

This note sets out the path from dated public sources to reviewed ratings:
- a model may *propose*;
- a person *reviews*;
- the engine *imports* only what the person accepted, and checks it.

## What already exists, and what is reused

| Existing piece | What it does | Fit |
| --- | --- | --- |
| External proposals (`docs/schemas/external-proposals-*.md`, `examples/external_proposals/`) | Typed proposals about **project cells** (alternative × criterion), reviewed as JSON events and compiled into a **separate project copy** | Not directly. Its subjects must be project alternatives and its output is a project; reference cases are neither. Reused here: a source registry; explicit per-proposal review statuses (accept, correct, reject; nothing is accepted by default); originals kept intact; a canonical digest; Python as the validation authority. |
| `scripts/rater_kit.py` | A blind second rater's packet (HTML), a CSV form, and conversion of answers | Reused here: the HTML packet with rubric text and labels; the spreadsheet-safe CSV form; reading answers as `utf-8-sig`. Not reused: it is tied to the study's cells and the classifier benchmark. |
| `build_rubrics` (`examples/nz_venture_screening/build.py`, written to `rubrics.json`) | Each scored leaf's rubric text and labels | Reused as is for the review page and the packets. |
| `reference-cases` version 2 | Ratings with `review: proposed / reviewed`, claims with dated sources, and the snapshot guard on partial dates | The import target. The format is unchanged. |
| The VEE bridge | Turns the VEE audit into a version 2 set with proposed cohort ratings | It supplies the cases. Packets read the same audit for each case's entry source. |

So the only new format is a small sidecar for **proposals about reference cases**. The rest is a
review sheet and an importer.

## 1. The flow

```text
v2 reference set + project (+ VEE audit, rubrics)  --packets-->       packet per case (no outcomes)
packets + public dated sources (outside the repo)   --proposer-->      rating-proposals v1 (in $MCDA_DATA)
proposals + project + set                           --review-sheet-->  review.csv (UTF-8, BOM) + review.html
a person edits review.csv: decision, final_label, reviewer_note
review.csv + proposals + set                        --import-reviews-->  updated v2 set (reviewed ratings)
updated set                                         --validate, outcomes-->  snapshot guard; coverage by review state
```

Commands:
- Engine and domain-neutral, as `mcda reference-cases` subcommands:
  - `packets` (the generic part);
  - `review-sheet`;
  - `import-reviews`.
- VEE-specific, in `examples/reference_rating/`:
  - `pilot.py sample` (the seeded case sample);
  - `pilot.py packets` (adds VEE's entry source and the rubric text to the generic packet).

### Packets (no outcome information)

A packet holds, for one case:
- the case id, name and snapshot date;
- context rows named on a **whitelist**;
- the leaves to rate, each with its labels (and rubric text when supplied).

It holds **no** outcome definitions, observations, outcome claims, outcome sources or
`observed_through`. Context rows are free text, and VEE's include an outcome status, so a context
row enters only when its label is whitelisted. A test asserts the packet's keys, recursively.

The VEE packet adds the entry source (title, publisher, URL and its date as VEE recorded it) from
the audit's `decisions[].entry`. It never reads `follow_up`.

## 2. The proposal record (`rating-proposals` version 1)

A separate JSON file with the strict sidecar conventions: unknown fields refused, `version` an
explicit integer, other versions refused.

- **Envelope:** `kind: "rating-proposals"`, `version: 1`, `reference_set_id`, `criteria_from` (the
  project id), `proposer` (`kind` human or tool, `proposed_by`, optional `tool_id` and
  `tool_version`: the evidence `Origin`), `created_on`, and `proposals`.
- **Proposal fields:**

  | Field | Rule |
  | --- | --- |
  | `id` | unique in the file |
  | `case_id` | a case of the reference set |
  | `criterion_id` | a labelled leaf of the project |
  | `label` | one of the leaf's labels; or `null` together with `not_enough_evidence: true` |
  | `claim` | 1 to 600 characters, in the proposer's own words |
  | `source_url` | a URL; required with a label |
  | `source_title` | optional |
  | `source_date` | a partial date (`YYYY`, `YYYY-Qn`, `YYYY-MM` or `YYYY-MM-DD`); required with a label |
  | `locator` | where in the source: a section, page or anchor; required with a label |
  | `excerpt` | at most 25 words, quoted from the source; required with a label |
  | `confidence` | `high`, `medium` or `low` |
  | `note` | optional |

- **"Not enough evidence"** is a proposal like any other. It may carry a source that was looked at.
  Accepting it leaves the leaf **unrated**: missing, not zero.
- **Structural refusals (whole file):**
  - a label that is not on the leaf's scale;
  - a label without a source URL, source date, locator or excerpt;
  - an excerpt over 25 words;
  - a duplicate id;
  - an unknown case or leaf.

## 3. The review sheet

openpyxl is not a pinned dependency and is not added. The sheet is a **UTF-8 CSV with a
byte-order mark**: Excel on a Korean (cp949) or any other locale then opens it as UTF-8. Every cell
that starts with `=`, `+`, `-` or `@` is prefixed with `'`, so Excel never evaluates it (as the
rater kit does).

**Columns, in order:**

`proposal_id, proposal_sha256, case_id, case_name, snapshot_date, criterion_id, criterion_name,
proposed_label, not_enough_evidence, allowed_labels, claim, source_url, source_date, locator,
excerpt, confidence, proposer, decision, final_label, reviewer_note`

- **The reviewer fills only `decision`, `final_label` and `reviewer_note`.**
  - `decision` is `accept`, `edit`, `reject` or `skip`. The sheet starts with every row `skip`.
  - The importer ignores every other column's text. It re-reads the proposal from the proposals
    file by `proposal_id`, and refuses a row whose `proposal_sha256` no longer matches.
- **Saving.** Save from Excel as *CSV UTF-8*. A sheet saved in another encoding is refused with
  that message, unless `--sheet-encoding` names its encoding.
- **The companion page `review.html`** lists, per leaf, the allowed labels in order with the
  rubric text, then the proposals grouped by case. Each proposal shows its source link, date,
  locator and excerpt. Keep it open beside the sheet.

## 4. Import rules (`import-reviews`)

**What each decision does:**

| `decision` | Effect |
| --- | --- |
| `accept` | The proposed label becomes a rating with `review: "reviewed"`. A "not enough evidence" proposal adds nothing: the leaf stays unrated. |
| `edit` | `final_label` becomes the rating, and it must be one of the leaf's labels (refused otherwise). The proposal must have its own dated source and excerpt, so an `edit` of a "not enough evidence" row without a source is refused: propose again with a source. |
| `reject`, `skip` | Nothing: the leaf stays unrated. |

**Per imported rating:**
- **The source** joins the set's `sources` under a key derived from the URL and date
  (`rp_` plus 12 hex characters of SHA-256). Its date is the proposal's `source_date` and its
  classification is `unknown`.
- **A claim** `rp_<proposal_id>` is added to the case. Its text is the proposer's claim plus the
  excerpt and locator, and its origin is the proposer.
- **The rating** cites that claim.
- **The note** reads "Reviewed by {reviewer} on {date}. {reviewer_note} (proposal {id},
  {confidence} confidence)". `--reviewer` and `--reviewed-on` are required arguments.

**Refusals (atomic: nothing is written, and every issue is listed):**
- an accepted or edited proposal whose source cannot support a rating at the case's snapshot. This
  is the version 2 snapshot guard: the latest day the date can mean must be on or before the
  snapshot;
- a missing source or excerpt;
- an unknown decision;
- a digest mismatch;
- a row whose proposal is not in the proposals file, or the same proposal twice;
- a second accepted proposal for the same case and leaf in one sheet.

**Existing ratings:**
- an existing **reviewed** rating on that case and leaf is never changed without `--replace`;
- an existing **proposed** rating (for example from the bridge) is replaced by the reviewed one.

**Idempotent:** importing the same sheet twice gives the same file. An identical claim, source and
rating already present is left as it is, and is not a conflict.

**Checked:** the result is validated with `reference_outcome_service.validate` before it is
written, so it always passes `mcda reference-cases validate`, snapshot guard included. The output
path must not exist (no overwrite).

## 5. Leakage controls, stated honestly

A proposer that is a language model may already know how a company turned out. Source dates
cannot remove that knowledge, so a proposal can be shaped by hindsight even when every cited
source predates the snapshot.

**What is controlled:**
- **The packet holds no outcome information.** The proposer is given the packet, the rubric and
  public sources, never the outcome files. In the pilot, the proposing subagent is given only the
  packets and the review page.
- **Every rating must cite a dated source and an excerpt**, and the snapshot guard refuses a
  source whose latest possible date is after the snapshot.
- **Proposals without a source or excerpt are refused** at import.
- **A person reviews every row.** Nothing is reviewed by default, and the importer counts only
  `accept` and `edit`.
- **Cases are chosen by a seeded random draw** over the whole entry cohort, failures included, and
  are never swapped for easier ones.

**What is not controlled:**
- **The proposer's prior knowledge** of later events, which can steer which sources it looks for,
  which excerpt it picks, and how it reads an ambiguous one.
- **Survivorship in the sources.** Companies that closed have fewer archived pages, so "not enough
  evidence" may correlate with the outcome. It is recorded as missing, which the calibration's
  missing-at-random assumption then has to bear.
- **Wayback and publication dates.** A capture date says when a page was archived, not when its
  text was written. A page can be backdated; a news item can be edited after publication.
- **The reviewer's own knowledge** of the outcomes.
- **Excerpt fidelity.** The importer checks that an excerpt exists and how long it is, not that it
  is on the page. The reviewer opens the source.

## 6. Scope and order

Rate first the scored leaves of the objective with the **highest stated weight across the NZ
screening profiles**, taking the mean over the two profiles: Technical/IP-led and
Commercial/traction-led.

| Objective | technical_ip | commercial_traction | Mean |
| --- | --- | --- | --- |
| Team | 0.300 | 0.250 | **0.275** |
| Problem and solution | 0.333 | 0.188 | 0.260 |
| Traction | 0.167 | 0.312 | 0.240 |
| Market | 0.200 | 0.250 | 0.225 |
| Mandate | 0 | 0 | 0 |

1. **Team:** founder–problem fit, team completeness, track record, board and advisors (all
   ordinal).
2. **Problem and solution:** problem severity, defensibility, readiness, regulatory burden.
3. **Traction**, then the labelled leaves of **Market** (competitive intensity, route to market).
   Market size and growth are numeric and are not rated through this flow.
4. **Mandate leaves last:** zero-weight gates that add nothing to calibration.

The pilot rates the first-priority leaves only (team), for a seeded random sample of 12 of the 55
cases.

## Verification

- `tests/unit/test_reference_rating.py`, on synthetic fixtures only:
  - every decision type;
  - an `edit` to an unknown label refused;
  - a proposal whose source is dated after the snapshot refused;
  - a proposal with no source or no excerpt refused;
  - an excerpt over 25 words refused;
  - re-import idempotent;
  - an existing reviewed rating protected without `--replace`;
  - the BOM CSV round-trip under cp949 and UTF-8;
  - packets free of outcome keys;
  - the project and the input set untouched;
  - the commands.
- The mutant group `reference_rating` in `scripts/mutation_check.py`.
