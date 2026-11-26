# VEE bridge: Venture Evidence Explorer cohort → reference set version 2

> **Real companies, public sources, VEE's own decisions.** The output names real New Zealand
> companies from Venture Evidence Explorer's (VEE) public research data. It is **not** committed
> here: write it to `$MCDA_DATA`. Nothing in it is investment advice or a prediction.

The bridge turns VEE's audited first institutional seed cohort into a
[version 2 reference set](../../docs/design/reference-cases.md#version-2-observation-aware-outcomes-proposed),
so Venture Frame can count outcomes without mistaking an unwatched company for a failure, and show
how much of its own rubric the cohort actually rates.

```sh
# 1. In a VEE checkout, produce VEE's own cohort audit (read-only):
python -c "import json; from backend.cohort_audit import run_audit; \
  json.dump(run_audit(), open('$MCDA_DATA/vee-cohort-audit.json', 'w'), default=str)"

# 2. Here:
python examples/vee_bridge/convert.py \
  --audit "$MCDA_DATA/vee-cohort-audit.json" \
  --snapshots path/to/vee/research/healthtech-pilot/backtest-snapshots.json \
  --project examples/nz_venture_screening/project.json \
  --out "$MCDA_DATA/vee-reference-cases.json"
mcda reference-cases outcomes "$MCDA_DATA/vee-reference-cases.json" \
  --project examples/nz_venture_screening/project.json
```

## What it does, and what it refuses to do

- **Outcomes come from VEE, never re-derived.** `reached_outcome` → `event`; closed or acquired
  before the outcome → `competing_event`; `observed_no_outcome` and `lost_to_follow_up` →
  `no_event` when watched through the window, otherwise `censored`; `unknown` → `unknown`. "Watched
  to" is the earlier of the first day of VEE's last verified activity and VEE's search date, which
  is VEE's own 36-month coverage rule.
- **Snapshot** = the last day VEE's entry date can mean (a month entry becomes its last day). The
  36-month window opens there.
- **Ratings** use `rubric-map.json`. Only exact equivalences by default; `--include-approximate`
  adds lower-bound mappings (for example VEE `prototype = yes` → Readiness *Prototype*), each marked
  in its note. Unmapped values stay unrated: missing, never zero. Every rating is `proposed`; a
  person reviews it before anything treats it as reviewed.
- A rating whose source is undated, or could be dated after the snapshot, is **dropped and
  counted**. The output never breaks the snapshot guard.

## What the first run showed (25 November 2026, VEE commit 1c13167, follow-up as of 2026-11-21)

| | Count |
| --- | --- |
| Cases (VEE included entries) | 55 |
| Event: verified priced Series A or later within 36 months | 2 |
| Competing event (closed before the outcome) | 1 |
| No event, watched through 36 months | 13 |
| Censored (watched for part of the window) | 6 |
| Unknown (no follow-up) | 33 |

The share among fully observed cases is 2 of 16 (12.5%, 95% interval 1.6% to 38.3%). Over all 55
it could be anywhere from 2/55 to 41/55 depending on the unresolved cases. These numbers match VEE's
own audit (55 entries, 2 verified, 33 unknown, 10 lost to follow-up, 13 covered at 36 months).

**Coverage is the binding constraint.** With exact mappings, the cohort rates 2 of the screening
study's 17 labelled leaves (HQ country for 17 cases and sector for 9, both mandate gates); with
approximate mappings, 4. None is reviewed. An outcome model cannot learn the screening weights from
this set until cohort companies are rated on the Venture Frame rubric at their snapshot dates.
