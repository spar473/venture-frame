"""The elicitation check and its proposals. Design: ``docs/design/structured-judgement.md``."""

from pathlib import Path

import pytest

from mcda_studio.domain.errors import UnknownReferenceError
from mcda_studio.domain.pairwise import PairwiseJudgement as J
from mcda_studio.domain.preferences import AnalysisConfiguration, Elicitation, GroupElicitation
from mcda_studio.methods.ahp.matrix import analyse
from mcda_studio.methods.smaa.validation import check_smaa
from mcda_studio.services import elicitation_service as es
from mcda_studio.services import project_service as ps
from mcda_studio.services import screening_robustness_service as robustness
from mcda_studio.storage.json_repository import JsonProjectRepository, to_json_text

STUDY = Path(__file__).parents[2] / "examples" / "nz_venture_screening" / "project.json"
TECH = "technical_ip"  # top level: team 90, problem_solution 100, market 60, traction 50, mandate 0
ORDER = ["problem_solution", "team", "market", "traction"]
CONSISTENT = [  # Saaty answers close to the swing ratios 100 : 90 : 60 : 50
    J(row="problem_solution", column="team", value=1),
    J(row="problem_solution", column="market", value=2),
    J(row="problem_solution", column="traction", value=2),
    J(row="team", column="market", value=2),
    J(row="team", column="traction", value=2),
    J(row="market", column="traction", value=1),
]


def study():  # type: ignore[no-untyped-def]
    return JsonProjectRepository().load(STUDY)


def recorded(conviction="medium", order=ORDER, pairwise=CONSISTENT, reasons=None):  # type: ignore[no-untyped-def]
    record = Elicitation(
        elicited_by="Study author",
        conviction=conviction,
        groups=[GroupElicitation(parent_id=None, order=order, pairwise=pairwise,
                                 reasons=reasons or {c: "why" for c in ORDER})],
    )  # fmt: skip
    return ps.set_elicitation(study(), TECH, record)


def top(check: es.ElicitationCheck) -> es.GroupCheck:
    return next(g for g in check.groups if g.parent_id is None)


def test_rank_order_centroid_weights() -> None:
    assert es.roc_weights(1) == [1.0]
    assert es.roc_weights(2) == pytest.approx([0.75, 0.25])
    assert es.roc_weights(4) == pytest.approx([25 / 48, 13 / 48, 7 / 48, 3 / 48])
    for n in range(1, 16):
        w = es.roc_weights(n)
        assert sum(w) == pytest.approx(1.0) and w == sorted(w, reverse=True)


def test_a_consistent_record_checks_clean_with_swing_roc_and_ahp_on_one_footing() -> None:
    check = es.elicitation_check(recorded(), TECH)
    group = top(check)
    assert check.recorded and check.conviction == "medium"
    assert group.status == "consistent" and group.order_agrees is True and not group.findings
    rows = {c.criterion_id: c for c in group.children}
    assert rows["mandate"].held and rows["mandate"].swing == 0.0
    assert rows["team"].swing == pytest.approx(90 / 300)
    # ROC for four ranked children; the mandate holds no weight, so the footing is the same
    assert rows["problem_solution"].roc == pytest.approx(25 / 48)
    assert rows["traction"].roc == pytest.approx(3 / 48)
    expected = analyse(ORDER, CONSISTENT)
    assert rows["team"].ahp == pytest.approx(expected.priorities[1])
    assert group.ahp is not None and group.ahp.complete
    assert group.ahp.consistency_ratio == pytest.approx(expected.consistency_ratio)
    assert group.largest_gap == pytest.approx(
        max(abs(rows[c].swing - rows[c].ahp) for c in ORDER)  # type: ignore[operator]
    )
    assert group.largest_gap < es.GAP_THRESHOLD and group.missing_reasons == []
    assert group.suggested_order == ORDER  # by swing rating; the mandate (rated 0) is left out
    assert group.scored and not next(g for g in check.groups if g.parent_id == "mandate").scored


def test_an_order_the_ratings_contradict_is_named() -> None:
    order = ["traction", "problem_solution", "team", "market"]  # traction is rated lowest
    group = top(es.elicitation_check(recorded(order=order, pairwise=[]), TECH))
    assert group.status == "check" and group.order_agrees is False
    assert ("traction", "problem_solution") in group.order_conflicts
    assert len(group.order_conflicts) == 3
    assert any("puts Traction above Problem and solution" in f for f in group.findings)
    assert es.order_conflicts(["a", "b"], {"a": 1.0, "b": 1.0}) == []  # ties never conflict


def test_inconsistent_or_incomplete_pairwise_answers_are_flagged() -> None:
    inconsistent = [*CONSISTENT[:5], J(row="traction", column="market", value=9)]
    group = top(es.elicitation_check(recorded(pairwise=inconsistent), TECH))
    assert group.ahp is not None and group.ahp.inconsistent and group.status == "check"
    assert group.ahp.consistency_ratio > 0.10 and len(group.ahp.triad) == 3
    group = top(es.elicitation_check(recorded(pairwise=CONSISTENT[:2]), TECH))
    assert group.ahp is not None and not group.ahp.complete and group.ahp.issues
    assert group.status == "check"


def test_a_large_swing_ahp_gap_asks_for_a_second_look() -> None:
    skewed = [J(row=a, column=b, value=9) for i, a in enumerate(ORDER) for b in ORDER[i + 1 :]]
    group = top(es.elicitation_check(recorded(pairwise=skewed), TECH))
    assert group.largest_gap is not None and group.largest_gap > es.GAP_THRESHOLD
    assert any("Swing and AHP weights differ" in f for f in group.findings)


def test_missing_reasons_and_an_incomplete_order_are_listed() -> None:
    group = top(es.elicitation_check(recorded(reasons={"team": "x"}), TECH))
    assert group.missing_reasons == ["problem_solution", "market", "traction"]
    group = top(es.elicitation_check(recorded(order=["team", "market"], pairwise=[]), TECH))
    assert group.status == "check" and group.order_agrees is None
    assert any("exactly once" in f for f in group.findings)


def test_missing_ratings_make_the_group_incomplete_with_no_proposals() -> None:
    project = ps.clear_judgement(recorded(), "team", TECH)
    check = es.elicitation_check(project, TECH)
    assert top(check).status == "incomplete" and check.status == "incomplete"
    assert all(p.smaa.groups for p in check.proposals)
    assert all(g.parent_id is not None for p in check.proposals for g in p.smaa.groups)


def test_conviction_proposal_bands_every_scored_group_and_holds_zero_ratings() -> None:
    project = recorded("high")
    check = es.elicitation_check(project, TECH)
    proposal = next(p for p in check.proposals if p.mode == "conviction")
    assert "±15%" in proposal.label
    groups = {g.parent_id: g for g in proposal.smaa.groups}
    assert "mandate" not in groups  # its branch carries no weight, so it is not varied
    assert set(groups) == {None, "team", "problem_solution", "market", "traction"}
    assert groups[None].held == ["mandate"] and groups[None].kind == "rating_intervals"
    assert groups[None].intervals["team"] == pytest.approx((90 * 0.85, 90 * 1.15))
    assert project.smaa is not None and proposal.smaa.cells == project.smaa.cells
    applied = ps.set_smaa_model(project, proposal.smaa)
    configuration = AnalysisConfiguration(method="smaa2", weight_profile_id=TECH)
    assert not check_smaa(applied, configuration, [])


def test_ranking_proposal_uses_the_stated_order_and_runs() -> None:
    project = recorded(conviction=None)
    check = es.elicitation_check(project, TECH)
    assert [p.mode for p in check.proposals] == ["ranking"]
    (group,) = check.proposals[0].smaa.groups
    assert group.kind == "ranking" and group.order == ORDER and group.held == ["mandate"]
    applied = ps.set_smaa_model(project, check.proposals[0].smaa)
    configuration = AnalysisConfiguration(method="smaa2", weight_profile_id=TECH)
    assert not check_smaa(applied, configuration, [])


def test_no_record_still_checks_ratings_and_proposes_nothing() -> None:
    check = es.elicitation_check(study(), TECH)
    assert not check.recorded and check.proposals == [] and check.status == "consistent"
    assert top(check).order == [] and top(check).order_agrees is None


def test_preview_gives_decision_confidence_now_and_under_each_proposal() -> None:
    project = recorded("medium")
    configuration = project.analysis.model_copy(update={"weight_profile_id": TECH})
    preview = es.elicitation_preview(project, configuration)
    assert [p.mode for p in preview.previews] == ["current", "conviction", "ranking"]
    assert preview.previews[0].robustness == robustness.screening_robustness(project, configuration)
    ranking = preview.previews[2].robustness
    assert ranking.available and "keeps the order" in ranking.basis
    by = {r.alternative_id: r for r in ranking.rows}
    assert by["upstock"].first.share < 0.6  # far less settled than under the typed intervals


def test_looking_changes_nothing_and_unknown_profiles_are_refused() -> None:
    project = recorded()
    before = to_json_text(project)
    es.elicitation_check(project, TECH)
    es.elicitation_preview(project, project.analysis)
    assert to_json_text(project) == before
    with pytest.raises(UnknownReferenceError):
        es.elicitation_check(project, "nobody")


def test_conviction_intervals_rule() -> None:
    assert es.conviction_intervals({"a": 100.0}, 0.3) == {"a": pytest.approx((70.0, 130.0))}
    assert es.CONVICTION_HALF_WIDTH == {"high": 0.15, "medium": 0.30, "low": 0.50}
