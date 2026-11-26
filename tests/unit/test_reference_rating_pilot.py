"""The VEE reference-rating pilot script (``examples/reference_rating/pilot.py``) on a synthetic
audit: a seeded sample that reads only case ids, packets without outcome or hindsight fields, and
outputs refused inside the repository.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import reference_case_builders as b
from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "reference_rating_pilot", ROOT / "examples" / "reference_rating" / "pilot.py"
)
assert spec is not None and spec.loader is not None
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

HINDSIGHT = "is the current trading name"


def v2_set(outcome_status: str = "event") -> ReferenceCasesV2:
    cases = []
    for case_id in ("k1", "k2", "k3", "k4", "k5"):
        raw = b.case(case_id, {})
        del raw["outcome"]
        raw["context"] = [
            {"label": "Entry round as reported", "value": "seed"},
            {"label": "VEE outcome status", "value": "reached_outcome"},
        ]
        raw["outcomes"] = [{"definition": "d", "status": "unknown"}]
        cases.append(raw)
    raw_set = b.reference_set(*cases)
    raw_set.update(
        version=2,
        observed_through="2026-11-20",
        outcome_definitions={"d": {"label": "D", "window_months": 36, "description": "S."}},
    )
    raw_set["cases"][0]["outcomes"] = [
        {
            "definition": "d",
            "status": outcome_status,
            "event_date": "2022-01-05",
            "claim_ids": ["c2"],
        }
        if outcome_status == "event"
        else {"definition": "d", "status": "unknown"}
    ]
    return ReferenceCasesV2.model_validate_json(json.dumps(raw_set))


def audit() -> dict[str, Any]:
    return {
        "sources": {
            "early": {"publisher": "Gazette", "url": "https://example.org/early",
                      "publication": {"value": "2021-06", "precision": "month"},
                      "scope": f"Example Ltd {HINDSIGHT}."},
            "late": {"publisher": "Annual report", "url": "https://example.org/late",
                     "publication": {"value": "2023-03-01", "precision": "day"}},
            "undated": {"publisher": "Blog", "url": "https://example.org/undated"},
        },
        "decisions": [
            {"company_id": "k1", "name": "K1 (renamed later)", "status": "included",
             "entry": {"company_id": "k1", "source_id": "late", "label": "seed funding",
                       "date": {"value": "2021-10", "precision": "month"}, "sector": "Tech",
                       "locator": "Annual Report 2023", "passage": f"K1 {HINDSIGHT}.",
                       "institutional_investor": "A fund",
                       "corroborating_source_ids": ["early", "late", "undated"]},
             "source_reports": [{"text": "later news"}]},
            {"company_id": "k2", "name": "K2", "status": "excluded", "entry": None},
        ],
        "follow_up": {"decisions": "reached_outcome for k1"},
    }  # fmt: skip


def test_the_sample_is_seeded_and_never_reads_outcomes() -> None:
    ids = [c.id for c in v2_set().cases]
    first = pilot.sample(ids, 3, 20261126)
    assert first == pilot.sample(list(reversed(ids)), 3, 20261126)  # order of input irrelevant
    assert len(set(first)) == 3 and set(first) <= set(ids)
    assert pilot.sample(ids, 3, 1) != first or pilot.sample(ids, 3, 2) != first
    # Changing outcomes cannot change the draw: only case ids go in.
    assert [c.id for c in v2_set("unknown").cases] == ids
    with pytest.raises(pilot.PilotError):
        pilot.sample(ids, 6, 0)


def test_vee_packets_withhold_outcomes_hindsight_and_later_sources() -> None:
    items = pilot.vee_packets(v2_set(), audit(), b.project(), {"scored_leaves": [
        {"id": "o4", "rubric": "Rubric O4."}]}, ["k1"], ["o4"])  # fmt: skip
    (packet,) = items
    text = json.dumps(packet)
    for leak in (HINDSIGHT, "reached_outcome", "VEE outcome status", "2022-01-05", "renamed later",
                 "later news", "Annual Report 2023", "https://example.org/late",
                 "https://example.org/undated", "2026-11-20"):  # fmt: skip
        assert leak not in text, leak
    assert packet["context"] == [{"label": "Entry round as reported", "value": "seed"}]
    entry = packet["vee_entry"]
    assert entry["entry"] == {"label": "seed funding", "date": {"value": "2021-10",
                              "precision": "month"}, "sector": "Tech",
                              "institutional_investor": "A fund"}  # fmt: skip
    assert entry["entry_source"] == {"source_id": "late", "usable_for_rating": False,
                                     "withheld": pilot.WITHHELD}  # fmt: skip
    usable = [s for s in entry["corroborating_sources"] if s["usable_for_rating"]]
    assert [s["url"] for s in usable] == ["https://example.org/early"]
    assert usable[0]["date"] == "2021-06"
    assert packet["leaves"][0]["rubric"] == "Rubric O4."


def test_outputs_inside_the_repository_are_refused(tmp_path: Path) -> None:
    with pytest.raises(pilot.PilotError, match="inside the repository"):
        pilot.write_json(ROOT / "examples" / "reference_rating" / "x.json", {})
    target = tmp_path / "out.json"
    pilot.write_json(target, {"a": "Māori"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": "Māori"}
    with pytest.raises(pilot.PilotError, match="overwrite"):
        pilot.write_json(target, {})


def test_rubric_text_reads_the_scored_leaves() -> None:
    rubrics = json.loads(
        (ROOT / "examples" / "nz_venture_screening" / "rubrics.json").read_text(encoding="utf-8")
    )
    text = pilot.rubric_text(rubrics)
    assert set(pilot.LEAF_GROUPS["team"]) <= set(text)
    assert "hq_country" not in text  # mandate leaves are not scored
