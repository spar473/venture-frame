"""The reference profile (A4): ordinal statements with citations, no numbers, never a default.

Design: docs/design/tradeoff-elicitation.md (Part 4)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcda_studio.services import judgement_aids_service as aids
from mcda_studio.storage.json_repository import JsonProjectRepository

from builders import flat_project

EXAMPLES = Path(__file__).parents[2] / "examples"
FILE = EXAMPLES / "reference_profiles" / "vc_literature.json"


def test_the_bundled_profile_is_ordinal_and_cited() -> None:
    (profile,) = aids.load_reference_profiles(FILE.parent)
    assert {v.id for v in profile.variants} == {"stated_by_vcs", "outcome_predictive"}
    urls = {c.url for v in profile.variants for c in v.citations}
    assert urls == {"https://www.nber.org/papers/w22587", "https://www.nber.org/papers/w33483"}
    text = FILE.read_text(encoding="utf-8")
    assert "%" not in text.replace("$1M", "").replace("$10M", "")  # no invented percentages
    raw = json.loads(text)
    for variant in raw["variants"]:
        for statement in variant["statements"]:
            assert set(statement) == {"horizon", "above", "below"}  # order only, no weights


def test_the_profile_is_strict() -> None:
    raw = json.loads(FILE.read_text(encoding="utf-8"))
    for change in ({"version": 2}, {"kind": "weights"}, {"weights": {"team": 0.5}}):
        with pytest.raises(ValidationError):
            aids.ReferenceProfile.model_validate_json(json.dumps({**raw, **change}))


def test_case_2_disagreements_follow_the_profiles_local_weights() -> None:
    project = JsonProjectRepository().load(EXAMPLES / "nz_venture_screening" / "project.json")
    (profile,) = aids.load_reference_profiles(FILE.parent)
    cfg = project.analysis.model_copy(update={"weight_profile_id": "commercial_traction"})
    result = aids.compare_reference(project, cfg, profile)
    assert result is not None
    # commercial_traction: team 80, problem_solution 60, market 80 (a tie with team)
    assert result.partner_order == ["team", "market", "problem_solution"]
    stated = next(v for v in result.variants if v.id == "stated_by_vcs")
    verdicts = {(p.above, p.below): p.partner for p in stated.statements[0].pairs}
    assert verdicts == {("team", "problem_solution"): "agrees", ("team", "market"): "tie"}


def test_a_project_without_the_criteria_gets_no_comparison() -> None:
    project = flat_project([[1.0, 0.0], [0.0, 1.0]], [60.0, 40.0])
    (profile,) = aids.load_reference_profiles(FILE.parent)
    assert aids.compare_reference(project, project.analysis, profile) is None


def test_anchors_run_from_the_worst_venture_to_the_best() -> None:
    project = JsonProjectRepository().load(EXAMPLES / "nz_venture_screening" / "project.json")
    found = {a.criterion_id: a for a in aids.anchors(project, project.analysis)}
    team = found["team_completeness"]
    assert (team.worst.name, team.worst.level) == ("Wych", "One key role named")
    assert (team.best.name, team.best.level) == ("Wellumio", "Next-stage leads named")
    assert team.swing == (
        "From Wych's level (One key role named) to Wellumio's (Next-stage leads named) on "
        "Team completeness"
    )
    for a in found.values():
        assert a.worst.value <= a.best.value
    assert "hq_country" not in found  # a leaf with no weight has no swing to picture


def test_pairs_in_different_sibling_groups_are_not_comparable() -> None:
    project = JsonProjectRepository().load(EXAMPLES / "nz_venture_screening" / "project.json")
    raw = json.loads(FILE.read_text(encoding="utf-8"))
    raw["variants"] = [raw["variants"][0] | {
        "statements": [
            {"horizon": None, "above": ["team_completeness"], "below": ["market", "track_record"]}
        ],
    }]  # fmt: skip
    profile = aids.ReferenceProfile.model_validate_json(json.dumps(raw))
    result = aids.compare_reference(project, project.analysis, profile)
    assert result is not None
    verdicts = {p.below: p.partner for p in result.variants[0].statements[0].pairs}
    # team_completeness (70) and track_record (50) are siblings; market is in another group
    assert verdicts == {"market": "not comparable", "track_record": "agrees"}
