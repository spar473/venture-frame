"""Convert a Venture Evidence Explorer (VEE) cohort audit into a version 2 reference set.

    python examples/vee_bridge/convert.py \
        --audit "$MCDA_DATA/vee-cohort-audit.json" \
        --snapshots path/to/vee/research/healthtech-pilot/backtest-snapshots.json \
        --project examples/nz_venture_screening/project.json \
        --out "$MCDA_DATA/vee-reference-cases.json"

``--audit`` is the JSON returned by VEE's own ``backend.cohort_audit.run_audit()``: this script
reads VEE's outcome decisions and never re-derives them. ``--snapshots`` is optional; without it
the cases carry outcomes and the cohort rating only.

Rules (README.md):

- The snapshot is the last day the entry date can mean, so nothing published later can support a
  rating, and the 36-month window opens no earlier than VEE's own latest horizon start.
- Every rating is ``proposed``. A person reviews it in Venture Frame before it counts as reviewed.
- Only exact rubric equivalences are used unless ``--include-approximate`` is passed.
- A rating whose source is undated or could be dated after the snapshot is dropped and counted;
  the output never breaks the snapshot guard.
- VEE statuses map one to one: reached_outcome -> event; closed or acquired before the outcome ->
  competing_event; observed_no_outcome and lost_to_follow_up -> no_event or censored by the date
  watched to (the earlier of the first day of VEE's last verified activity and its search
  date, as in VEE's own 36-month coverage rule); unknown -> unknown.

Venture-specific by design, so it lives with the examples, not in the engine. Write the output
outside the repository (``$MCDA_DATA``): it is real, public-source data.
"""

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2, add_months, date_bounds

DEFINITION = "series_a_plus_36m"
WINDOW = 36
ORIGIN = {
    "kind": "tool",
    "proposed_by": "VEE bridge (examples/vee_bridge)",
    "tool_id": "vee_bridge",
    "tool_version": "1",
}


def partial(date: dict[str, Any] | None) -> str | None:
    """VEE's {value, precision} as a partial date, or None when it is unknown."""
    if not date or date.get("value") is None or date.get("precision") == "unknown":
        return None
    return str(date["value"])


def source_date(source: dict[str, Any]) -> str | None:
    if source.get("published_on"):
        return str(source["published_on"])
    return partial(source.get("publication"))


def convert_source(key: str, source: dict[str, Any]) -> dict[str, Any]:
    publisher = source.get("publisher") or "Unknown publisher"
    kind = str(source.get("kind", "source")).replace("_", " ")
    retrieved = source.get("retrieved_on", "an unrecorded date")
    out: dict[str, Any] = {
        "name": source.get("title") or publisher,
        "reference": source.get("url") or f"vee:{key}",
        "location": publisher,
        "description": f"VEE {kind} from {publisher}, retrieved {retrieved}.",
        "classification": "unknown",
    }
    date = source_date(source)
    if date is not None:
        date_bounds(date)  # refuse a form the reference set cannot read
        out["date"] = date
    return out


class Case:
    def __init__(self, entry: dict[str, Any]) -> None:
        self.entry = entry
        first_last = date_bounds(partial(entry["date"]) or "")
        self.snapshot: dt.date = first_last[1]
        self.claims: list[dict[str, Any]] = []
        self.ratings: list[dict[str, Any]] = []

    def claim(self, claim_id: str, text: str, source: str) -> str:
        self.claims.append(
            {"id": claim_id, "text": text[:2000], "source": source, "origin": ORIGIN}
        )
        return claim_id


def _rateable(source_id: str, sources: dict[str, Any], snapshot: dt.date) -> bool:
    source = sources.get(source_id)
    if source is None or "date" not in source:
        return False
    return date_bounds(source["date"])[1] <= snapshot


def build(
    audit: dict[str, Any],
    snapshots: dict[str, Any] | None,
    rubric: dict[str, Any],
    project_id: str,
    include_approximate: bool,
) -> tuple[dict[str, Any], Counter[str]]:
    follow_up = audit["follow_up"]
    raw_sources = {**audit["sources"], **follow_up["sources"]}
    profiles: dict[str, dict[str, Any]] = {}
    if snapshots is not None:
        for source in snapshots["sources"]:
            raw_sources.setdefault(source["source_id"], source)
        profiles = {p["company_id"]: p for p in snapshots["profiles"]}
    sources = {key: convert_source(key, value) for key, value in raw_sources.items()}
    use = [m for m in rubric["features"] if m["exact"] or include_approximate]
    tally: Counter[str] = Counter()
    cases = []
    for row in audit["decisions"]:
        if row["status"] != "included":
            continue
        entry = row["entry"]
        if partial(entry["date"]) is None:
            tally["skipped: entry date unknown"] += 1
            continue
        case = Case(entry)
        for rule in rubric["cohort"]:
            if not rule["exact"] and not include_approximate:
                continue
            if _rateable(entry["source_id"], sources, case.snapshot):
                claim = case.claim(
                    f"cohort_{rule['criterion_id']}", rule["rationale"], entry["source_id"]
                )
                case.ratings.append(_rating(rule, claim))
                tally["ratings"] += 1
            else:
                tally["dropped: entry source undated or after the snapshot"] += 1
        profile = profiles.get(entry["company_id"])
        for feature in profile["features"] if profile else []:
            for rule in use:
                if (feature["name"], str(feature["value"])) != (rule["feature"], rule["value"]):
                    continue
                if any(r["criterion_id"] == rule["criterion_id"] for r in case.ratings):
                    tally["dropped: criterion already rated"] += 1
                elif _rateable(feature["source_id"], sources, case.snapshot):
                    text = f"VEE {feature['name']} = {feature['value']}. {feature['rationale']}"
                    claim = case.claim(f"feature_{feature['name']}", text, feature["source_id"])
                    case.ratings.append(_rating(rule, claim))
                    tally["ratings"] += 1
                else:
                    tally["dropped: feature source undated or after the snapshot"] += 1
        decision = follow_up["decisions"][entry["company_id"]]
        outcome = _outcome(case, decision, follow_up)
        tally[f"outcome: {outcome['status']}"] += 1
        cases.append(
            {
                "id": entry["company_id"],
                "name": row["name"],
                "snapshot_date": case.snapshot.isoformat(),
                "context": [
                    {"label": "Entry round as reported", "value": entry["label"]},
                    {
                        "label": "Entry date as reported",
                        "value": partial(entry["date"]) or "unknown",
                    },
                    {"label": "Sector as reported", "value": entry["sector"]},
                    {"label": "VEE outcome status", "value": decision["status"]},
                ],
                "claims": case.claims,
                "ratings": case.ratings,
                "outcomes": [outcome],
            }
        )
    reference_set = {
        "kind": "reference-cases",
        "version": 2,
        "id": "vee_nz_seed_cohort",
        "title": "VEE New Zealand first institutional seed cohort",
        "description": (
            f"Converted from VEE's cohort audit ({audit['rule']}, evaluated "
            f"{audit['evaluated_on']}, follow-up as of {follow_up['as_of']}). Ratings are "
            "proposed, not reviewed."
        ),
        "criteria_from": project_id,
        "observed_through": follow_up["as_of"],
        "outcome_definitions": {
            DEFINITION: {
                "label": "Verified priced Series A or later within 36 months",
                "window_months": WINDOW,
                "description": (
                    "VEE's primary endpoint: a priced lettered round with new company equity, "
                    "distinct from the entry round and proved by VEE's next-round evidence."
                ),
            }
        },
        "cases": cases,
        "sources": sources,
    }
    return reference_set, tally


def _rating(rule: dict[str, Any], claim: str) -> dict[str, Any]:
    note = "" if rule["exact"] else f"Approximate: {rule['rationale']}"
    return {
        "criterion_id": rule["criterion_id"],
        "label": rule["label"],
        "claim_ids": [claim],
        "review": "proposed",
        "note": note,
    }


def _outcome(case: Case, decision: dict[str, Any], follow_up: dict[str, Any]) -> dict[str, Any]:
    status = decision["status"]
    out: dict[str, Any] = {"definition": DEFINITION}
    events = {e["event_id"]: e for e in decision["events"]}
    if status in ("reached_outcome", "closed_before_outcome", "acquired_or_ipo_before_outcome"):
        event = events[decision["outcome_event_id"]]
        claim = case.claim(f"outcome_{event['event_id']}", event["claim"], event["source_id"])
        out.update(
            status="event" if status == "reached_outcome" else "competing_event",
            event_date=partial(event["date"]),
            claim_ids=[claim],
            note="" if status == "reached_outcome" else f"VEE: {status} ({event['kind']}).",
        )
        return out
    review = decision.get("review")
    if status in ("observed_no_outcome", "lost_to_follow_up") and review and decision["censor_on"]:
        # VEE's conservative rule: a case counts as watched only from the first day its last
        # verified activity can mean, and no further than the search went.
        activity = events.get(review.get("activity_event_id") or "")
        seen = date_bounds(partial(activity["date"]) or "")[0] if activity else None
        watched = min(
            seen or dt.date.fromisoformat(decision["censor_on"]),
            dt.date.fromisoformat(review["searched_through"]),
        )
        end = add_months(case.snapshot, WINDOW)
        if watched >= case.snapshot:
            claims = []
            if review["source_ids"]:
                claims = [case.claim("outcome_review", review["scope"], review["source_ids"][0])]
            out.update(
                status="no_event" if watched >= end else "censored",
                observed_through=watched.isoformat(),
                claim_ids=claims,
                note=f"VEE: {status}.",
            )
            return out
    out.update(status="unknown", note=f"VEE: {status}.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--snapshots", type=Path)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=Path(__file__).with_name("rubric-map.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-approximate", action="store_true")
    args = ap.parse_args(argv)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    project_id = read(args.project)["id"]
    rubric = read(args.map)
    if rubric["criteria_from"] != project_id:
        print(
            f"the rubric map is for {rubric['criteria_from']!r}, not {project_id!r}",
            file=sys.stderr,
        )
        return 2
    snapshots = read(args.snapshots) if args.snapshots else None
    reference_set, tally = build(
        read(args.audit), snapshots, rubric, project_id, args.include_approximate
    )
    ReferenceCasesV2.model_validate_json(json.dumps(reference_set))  # the format, before writing
    args.out.write_text(
        json.dumps(reference_set, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for key, count in sorted(tally.items()):
        print(f"{count:4d}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
