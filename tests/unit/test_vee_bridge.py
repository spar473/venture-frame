"""The VEE bridge (examples/vee_bridge): a small synthetic audit shaped like VEE's cohort audit is
converted into a version 2 reference set that passes the engine's validation, with every VEE status
mapped as the README states, every rating proposed, and no rating that breaks the snapshot guard.
No real data is used here; the real conversion runs against $MCDA_DATA outside the repository."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2
from mcda_studio.services import reference_outcome_service as svc
from mcda_studio.storage.json_repository import JsonProjectRepository

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "examples/vee_bridge"
PROJECT = ROOT / "examples/nz_venture_screening/project.json"

spec = importlib.util.spec_from_file_location("vee_bridge_convert", BRIDGE / "convert.py")
assert spec and spec.loader
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def day(value: str) -> dict[str, str]:
    return {"value": value, "precision": "day"}


def month(value: str) -> dict[str, str]:
    return {"value": value, "precision": "month"}


def entry(company: str, date: dict[str, Any], source: str = "entry_src") -> dict[str, Any]:
    return {
        "company_id": company,
        "name": f"{company} (trading name)",
        "status": "included",
        "entry": {
            "company_id": company,
            "source_id": source,
            "sector": "Software",
            "date": date,
            "label": "seed",
        },
    }


def event(event_id: str, kind: str, date: dict[str, Any], source: str = "follow_src") -> dict:
    return {
        "event_id": event_id,
        "kind": kind,
        "date": date,
        "source_id": source,
        "claim": f"Synthetic {kind}.",
    }


def review(searched_through: str, activity: str | None) -> dict[str, Any]:
    return {
        "searched_through": searched_through,
        "scope": "Synthetic review scope.",
        "source_ids": ["follow_src"],
        "activity_event_id": activity,
    }


def audit() -> dict[str, Any]:
    decisions = {
        # entry 2019-03 -> snapshot 2019-03-31, window ends 2022-03-31
        "hit": {"status": "reached_outcome", "outcome_event_id": "hit_a", "censor_on": None,
                "events": [event("hit_a", "series_a", month("2021-06"))], "review": None},
        "closed": {"status": "closed_before_outcome", "outcome_event_id": "closed_liq",
                   "censor_on": None,
                   "events": [event("closed_liq", "closed_liquidation", day("2020-02-02"))],
                   "review": None},
        "watched": {"status": "observed_no_outcome", "outcome_event_id": None,
                    "censor_on": "2026-06-30",
                    "events": [event("w_act", "last_verified_activity", month("2026-06"))],
                    "review": review("2026-11-20", "w_act")},
        "quiet": {"status": "lost_to_follow_up", "outcome_event_id": None,
                  "censor_on": "2021-08-31",
                  "events": [event("q_act", "last_verified_activity", month("2021-08"))],
                  "review": review("2026-11-20", "q_act")},
        # the activity month straddles the window end, so VEE leaves coverage open: censored
        "edge": {"status": "lost_to_follow_up", "outcome_event_id": None,
                 "censor_on": "2022-03-31",
                 "events": [event("e_act", "last_verified_activity", month("2022-03"))],
                 "review": review("2026-11-20", "e_act")},
        "dark": {"status": "unknown", "outcome_event_id": None, "censor_on": None,
                 "events": [], "review": None},
    }  # fmt: skip
    rows = [entry(company, month("2019-03")) for company in decisions]
    rows.append({"company_id": "out", "name": "Out", "status": "unresolved", "entry": None})
    return {
        "rule": "synthetic:1",
        "evaluated_on": "2026-11-20",
        "decisions": rows,
        "sources": {
            "entry_src": {"publisher": "P", "url": "https://example.org/e", "retrieved_on": "x",
                          "publication": month("2019-02")},
        },
        "follow_up": {
            "as_of": "2026-11-21",
            "decisions": decisions,
            "sources": {
                "follow_src": {"source_id": "follow_src", "title": "Later", "publisher": "Q",
                               "url": "https://example.org/f", "kind": "news_article",
                               "published_on": None, "retrieved_on": "2026-11-21"},
            },
        },
    }  # fmt: skip


SNAPSHOTS = {
    "sources": [
        {"source_id": "early", "title": "Early", "publisher": "R", "url": "https://example.org/r",
         "kind": "news_article", "published_on": "2019-03-15", "retrieved_on": "2026-11-22"},
        {"source_id": "late", "title": "Late", "publisher": "R", "url": "https://example.org/l",
         "kind": "news_article", "published_on": "2019-04-02", "retrieved_on": "2026-11-22"},
    ],
    "profiles": [
        {"company_id": "hit", "features": [
            {"name": "sector_normalised", "value": "healthtech", "rationale": "r.",
             "source_id": "early"},
            {"name": "prototype", "value": "yes", "rationale": "r.",
             "source_id": "early"},
            {"name": "buyer", "value": "business", "rationale": "r.",
             "source_id": "early"},
        ]},
        {"company_id": "watched", "features": [
            {"name": "sector_normalised", "value": "fintech", "rationale": "r.",
             "source_id": "late"},
        ]},
    ],
}  # fmt: skip


def convert(include_approximate: bool = False) -> tuple[dict[str, Any], Any]:
    rubric = json.loads((BRIDGE / "rubric-map.json").read_text(encoding="utf-8"))
    return bridge.build(audit(), SNAPSHOTS, rubric, rubric["criteria_from"], include_approximate)


@pytest.fixture(scope="module")
def project() -> Any:
    return JsonProjectRepository().load(PROJECT)


def test_the_output_is_a_valid_v2_set_for_the_screening_project(project: Any) -> None:
    raw, _ = convert()
    report = svc.validate(project, ReferenceCasesV2.model_validate_json(json.dumps(raw)))
    assert report.valid, report.issues
    assert [c["id"] for c in raw["cases"]] == ["hit", "closed", "watched", "quiet", "edge", "dark"]


def test_every_vee_status_maps_as_documented() -> None:
    raw, tally = convert()
    got = {c["id"]: c["outcomes"][0] for c in raw["cases"]}
    assert {k: v["status"] for k, v in got.items()} == {
        "hit": "event",
        "closed": "competing_event",
        "watched": "no_event",
        "quiet": "censored",
        "edge": "censored",
        "dark": "unknown",
    }
    assert got["hit"]["event_date"] == "2021-06"
    assert got["closed"]["event_date"] == "2020-02-02" and "closed" in got["closed"]["note"]
    assert got["watched"]["observed_through"] == "2026-06-01"  # first day of the activity month
    assert got["quiet"]["observed_through"] == "2021-08-01"
    assert got["edge"]["observed_through"] == "2022-03-01"
    assert all(c["snapshot_date"] == "2019-03-31" for c in raw["cases"])
    assert tally["outcome: censored"] == 2


def test_ratings_are_proposed_exact_only_and_never_break_the_snapshot_guard() -> None:
    raw, tally = convert()
    ratings = {c["id"]: {r["criterion_id"]: r for r in c["ratings"]} for c in raw["cases"]}
    assert set(ratings["hit"]) == {
        "hq_country",
        "sector",
    }  # prototype is approximate; buyer unmapped
    assert ratings["hit"]["sector"]["label"] == "Health and medtech"
    assert "sector" not in ratings["watched"]  # its source is dated after the snapshot
    assert tally["dropped: feature source undated or after the snapshot"] == 1
    assert all(r["review"] == "proposed" for c in ratings.values() for r in c.values())


def test_approximate_mappings_only_on_request_and_marked() -> None:
    raw, _ = convert(include_approximate=True)
    readiness = next(r for r in raw["cases"][0]["ratings"] if r["criterion_id"] == "readiness")
    assert readiness["label"] == "Prototype" and readiness["note"].startswith("Approximate:")


def test_the_summary_on_the_converted_set(project: Any) -> None:
    raw, _ = convert()
    result = svc.reference_outcomes(project, ReferenceCasesV2.model_validate_json(json.dumps(raw)))
    (summary,) = result.outcomes
    assert (summary.fully_observed, summary.observed_share and summary.observed_share.exact) == (
        3,
        "1/3",
    )
    assert summary.highest_share and summary.highest_share.exact == "2/3"  # 6 - 1 - 1 of 6
    assert result.reviewed_rating_count == 0
