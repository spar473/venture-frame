r"""Reference rating pilot for the VEE cohort (``docs/design/reference-rating.md``).

Venture-specific, so it lives with the examples. Every output is real, public-source data about real
companies: write it outside the repository (``$MCDA_DATA``); the script refuses paths inside it.

    python examples/reference_rating/pilot.py sample \
        --reference-cases "$MCDA_DATA/vee-reference-cases.json" --size 12 --seed 20261126 \
        --out "$MCDA_DATA/dd6/pilot-cases.json"
    python examples/reference_rating/pilot.py rubric-text \
        --rubrics examples/nz_venture_screening/rubrics.json --out "$MCDA_DATA/dd6/rubric-text.json"
    python examples/reference_rating/pilot.py packets \
        --reference-cases "$MCDA_DATA/vee-reference-cases.json" \
        --audit "$MCDA_DATA/vee-cohort-audit.json" \
        --project examples/nz_venture_screening/project.json \
        --rubrics examples/nz_venture_screening/rubrics.json \
        --cases "$MCDA_DATA/dd6/pilot-cases.json" --leaves team --out "$MCDA_DATA/dd6/packets"

``sample`` draws cases from the reference set's case ids only (sorted), with
``numpy.random.default_rng(seed)``: it never reads an outcome, so it cannot select by one.
``packets`` adds VEE's entry source (title, publisher, URL, date as VEE recorded it) from the
audit's ``decisions[].entry`` to the engine's packet; it never reads the audit's follow-up section,
and context rows enter only from a whitelist that excludes VEE's outcome status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2, date_bounds
from mcda_studio.services import reference_rating_service as rating
from mcda_studio.storage.json_repository import JsonProjectRepository

ROOT = Path(__file__).resolve().parents[2]
#: Context rows a packet may carry. VEE's "VEE outcome status" row is deliberately absent.
CONTEXT_WHITELIST = ["Entry round as reported", "Entry date as reported", "Sector as reported"]
#: The leaves to rate first: the team objective (highest mean weight over the two NZ profiles).
LEAF_GROUPS = {
    "team": ["founder_problem_fit", "team_completeness", "track_record", "board_advisors"],
    "problem_solution": ["problem_severity", "defensibility", "readiness", "regulatory_burden"],
    "traction": ["customer_validation", "commercial_stage", "partnerships", "milestone_delivery"],
    "market": ["competitive_intensity", "route_to_market"],
}
#: Audit entry fields a packet may show. VEE's own commentary (``passage``, the display ``name``,
#: a source's ``scope``) was written later and can carry hindsight, so it is left out, as is
#: anything from the follow-up section.
ENTRY_FIELDS = ["label", "date", "sector", "institutional_investor", "locator"]
SOURCE_FIELDS = ["publisher", "title", "url", "kind"]
WITHHELD = "dated after the snapshot, or undated: withheld so it cannot show later history"


class PilotError(ValueError):
    pass


def outside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise PilotError(f"{path} is inside the repository; real data stays in $MCDA_DATA")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    outside_repo(path)
    if path.exists():
        raise PilotError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sample(case_ids: list[str], size: int, seed: int) -> list[str]:
    """A seeded draw without replacement from the sorted case ids; nothing else is read."""
    ordered = sorted(case_ids)
    if not 1 <= size <= len(ordered):
        raise PilotError(f"cannot draw {size} of {len(ordered)} cases")
    picks = np.random.default_rng(seed).choice(len(ordered), size=size, replace=False)
    return [ordered[int(i)] for i in picks]


def rubric_text(rubrics: dict[str, Any]) -> dict[str, str]:
    return {str(leaf["id"]): str(leaf["rubric"]) for leaf in rubrics.get("scored_leaves", [])}


def _source_date(source: dict[str, Any]) -> str | None:
    if source.get("published_on"):
        return str(source["published_on"])
    publication = source.get("publication") or {}
    if publication.get("value") is None or publication.get("precision") == "unknown":
        return None
    return str(publication["value"])


def _pointer(sources: dict[str, Any], source_id: str, snapshot: str) -> dict[str, Any]:
    """A source as a pointer: who, where and when, and whether it can support a rating."""
    source = sources.get(source_id, {})
    date = _source_date(source)
    usable = date is not None and date_bounds(date)[1].isoformat() <= snapshot
    if not usable:
        # A source dated after the snapshot shows the case still existed then: follow-up
        # information. Withhold it; an undated one might be later, so it is withheld too.
        return {"source_id": source_id, "usable_for_rating": False, "withheld": WITHHELD}
    out: dict[str, Any] = {k: source[k] for k in SOURCE_FIELDS if k in source}
    out["source_id"] = source_id
    out["date"] = date
    out["usable_for_rating"] = True
    return out


def entry_sources(audit: dict[str, Any], snapshots: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Each included case's entry record and sources, by case id, without any follow-up field."""
    sources = audit.get("sources", {})
    out: dict[str, dict[str, Any]] = {}
    for row in audit.get("decisions", []):
        if row.get("status") != "included" or not row.get("entry"):
            continue
        entry = row["entry"]
        case_id = str(entry["company_id"])
        if case_id not in snapshots:
            continue
        snapshot = snapshots[case_id]
        entry_source = _pointer(sources, str(entry.get("source_id")), snapshot)
        fields = [k for k in ENTRY_FIELDS if k != "locator" or entry_source["usable_for_rating"]]
        out[case_id] = {
            "entry": {k: entry[k] for k in fields if k in entry},
            "entry_source": entry_source,
            "corroborating_sources": [
                _pointer(sources, str(s), snapshot)
                for s in entry.get("corroborating_source_ids", [])
            ],
        }
    return out


def vee_packets(
    reference_set: ReferenceCasesV2,
    audit: dict[str, Any],
    project: Any,
    rubrics: dict[str, Any],
    case_ids: list[str],
    leaves: list[str],
) -> list[dict[str, Any]]:
    base = rating.packets(
        project, reference_set, case_ids, leaves, CONTEXT_WHITELIST, rubric_text(rubrics)
    )
    snapshots = {p.case_id: p.snapshot_date.isoformat() for p in base}
    entries = entry_sources(audit, snapshots)
    out = []
    for packet in rating.dump_packets(base):
        packet["vee_entry"] = entries.get(packet["case_id"], {})
        out.append(packet)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--reference-cases", type=Path, required=True)
    s.add_argument("--size", type=int, default=12)
    s.add_argument("--seed", type=int, default=20261126)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("rubric-text")
    r.add_argument("--rubrics", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("packets")
    p.add_argument("--reference-cases", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--leaves", choices=sorted(LEAF_GROUPS), default="team")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "sample":
            reference_set = ReferenceCasesV2.model_validate_json(
                args.reference_cases.read_text(encoding="utf-8")
            )
            ids = [case.id for case in reference_set.cases]
            picked = sample(ids, args.size, args.seed)
            write_json(
                args.out,
                {"seed": args.seed, "size": args.size, "population": len(ids),
                 "method": "numpy.random.default_rng(seed).choice over sorted case ids, "
                           "without replacement", "cases": picked},
            )  # fmt: skip
            print("\n".join(picked))
        elif args.command == "rubric-text":
            write_json(args.out, rubric_text(read_json(args.rubrics)))
        else:
            reference_set = ReferenceCasesV2.model_validate_json(
                args.reference_cases.read_text(encoding="utf-8")
            )
            project = JsonProjectRepository().load(args.project)
            chosen = read_json(args.cases)["cases"]
            items = vee_packets(
                reference_set, read_json(args.audit), project, read_json(args.rubrics), chosen,
                LEAF_GROUPS[args.leaves],
            )  # fmt: skip
            outside_repo(args.out)
            for item in items:
                write_json(args.out / f"{item['case_id']}.json", item)
            print(f"wrote {len(items)} packets to {args.out}")
    except (PilotError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
