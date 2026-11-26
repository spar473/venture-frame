"""Schema version 7: elicitation records, ranking weight groups and held children.

A project that uses none of them keeps its bytes, fingerprint and version; one that does is
version 7; a version-6 file cannot hold them. Design: ``docs/design/structured-judgement.md``.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from mcda_studio.api import operations as ops
from mcda_studio.domain.errors import ProjectLoadError
from mcda_studio.domain.pairwise import PairwiseJudgement
from mcda_studio.domain.preferences import AnalysisConfiguration, Elicitation, GroupElicitation
from mcda_studio.domain.project import SCHEMA_VERSION, STRUCTURED_JUDGEMENT_VERSION
from mcda_studio.domain.uncertainty import GroupUncertainty, SmaaModel
from mcda_studio.methods.smaa.validation import check_smaa
from mcda_studio.services import project_service as ps
from mcda_studio.storage.json_repository import JsonProjectRepository, to_json_text

ROOT = Path(__file__).parents[2]
STUDY = ROOT / "examples" / "nz_venture_screening" / "project.json"
repo = JsonProjectRepository()
SMAA = AnalysisConfiguration(method="smaa2", weight_profile_id="technical_ip")
TOP = ["team", "problem_solution", "market", "traction"]


def record() -> Elicitation:
    return Elicitation(
        elicited_by="Study author",
        elicited_on=date(2026, 11, 25),
        conviction="medium",
        note="Practice elicitation on public information",
        groups=[
            GroupElicitation(
                parent_id=None,
                order=["problem_solution", "team", "market", "traction"],
                pairwise=[PairwiseJudgement(row="problem_solution", column="team", value=1)],
                reasons={"team": "Most of what can be verified at seed"},
            )
        ],
    )


def ranking(**extra: object) -> GroupUncertainty:
    return GroupUncertainty(
        parent_id=None,
        kind="ranking",
        order=["problem_solution", "team", "market", "traction"],
        held=["mandate"],
        **extra,  # type: ignore[arg-type]
    )


def issues(project, configuration=SMAA):  # type: ignore[no-untyped-def]
    return {(i.code, i.path) for i in check_smaa(project, configuration, [])}


def test_schema_version_is_7_and_unused_fields_change_nothing(tmp_path: Path) -> None:
    # Structured judgement arrived in version 7; version 8 (samples weight source) came later.
    assert STRUCTURED_JUDGEMENT_VERSION == 7 and SCHEMA_VERSION >= 7
    project = repo.load(STUDY)
    assert project.schema_version == project.required_schema_version() == 6
    assert repo.save(project, tmp_path / "p.json").read_bytes() == STUDY.read_bytes()
    assert "elicitation" not in to_json_text(project)
    assert project.smaa is not None
    for group in project.smaa.model_dump(mode="json")["groups"]:
        assert "order" not in group and "held" not in group


def test_an_elicitation_record_makes_version_7_and_round_trips(tmp_path: Path) -> None:
    project = ps.set_elicitation(repo.load(STUDY), "technical_ip", record())
    assert project.required_schema_version() == 7
    path = repo.save(project, tmp_path / "p.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 7
    saved = raw["weight_profiles"][0]["elicitation"]
    assert saved["elicited_on"] == "2026-11-25" and saved["conviction"] == "medium"
    assert repo.load(path) == project
    cleared = ps.set_elicitation(project, "technical_ip", None)
    assert to_json_text(cleared) == STUDY.read_text(encoding="utf-8")


def test_a_ranking_group_or_held_children_make_version_7() -> None:
    project = repo.load(STUDY)
    assert project.smaa is not None
    for group in [ranking(), GroupUncertainty(parent_id=None, kind="unknown", held=["mandate"])]:
        model = project.smaa.model_copy(update={"groups": [group]})
        assert ps.set_smaa_model(project, model).required_schema_version() == 7


@pytest.mark.parametrize(
    ("where", "value"),
    [
        ("profile", {"elicited_by": "x", "groups": []}),
        ("group", {"parent_id": None, "kind": "ranking", "order": TOP}),
        ("group", {"parent_id": None, "kind": "ranking"}),
        ("group", {"parent_id": None, "kind": "unknown", "held": ["mandate"]}),
    ],
)
def test_a_version_6_file_cannot_hold_version_7_fields(where: str, value: dict) -> None:  # type: ignore[type-arg]
    raw = json.loads(STUDY.read_text(encoding="utf-8"))
    if where == "profile":
        raw["weight_profiles"][0]["elicitation"] = value
    else:
        raw["smaa"]["groups"] = [value]
    with pytest.raises(ProjectLoadError, match="schema version 6 cannot contain"):
        repo.from_dict(raw)


def test_the_set_elicitation_operation_records_and_clears() -> None:
    project = repo.load(STUDY)
    edited = ops.apply_operation(
        project, ops.SetElicitation(profile_id="commercial_traction", elicitation=record())
    )
    by = {p.id: p for p in edited.weight_profiles}
    assert by["commercial_traction"].elicitation == record()
    assert by["technical_ip"].elicitation is None
    assert (
        by["commercial_traction"].judgements
        == {p.id: p for p in project.weight_profiles}["commercial_traction"].judgements
    )  # ratings are never changed
    back = ops.apply_operation(
        edited, ops.SetElicitation(profile_id="commercial_traction", elicitation=None)
    )
    assert to_json_text(back) == to_json_text(project)


def test_removing_a_criterion_leaves_every_record_and_group() -> None:
    project = ps.set_elicitation(repo.load(STUDY), "technical_ip", record())
    assert project.smaa is not None
    project = ps.set_smaa_model(project, project.smaa.model_copy(update={"groups": [ranking()]}))
    out = ps.remove_criterion(project, "team")
    group = next(g for g in out.smaa.groups if g.parent_id is None)  # type: ignore[union-attr]
    assert "team" not in group.order and group.held == ["mandate"]
    kept = out.weight_profiles[0].elicitation
    assert kept is not None
    assert kept.groups[0].order == ["problem_solution", "market", "traction"]
    assert kept.groups[0].pairwise == [] and kept.groups[0].reasons == {}
    out = ps.remove_criterion(out, "mandate")
    group = next(g for g in out.smaa.groups if g.parent_id is None)  # type: ignore[union-attr]
    assert group.held == []


def test_validation_of_ranking_groups_and_held_children() -> None:
    project = repo.load(STUDY)
    assert project.smaa is not None

    def with_group(group: GroupUncertainty):  # type: ignore[no-untyped-def]
        return ps.set_smaa_model(project, SmaaModel(cells=project.smaa.cells, groups=[group]))  # type: ignore[union-attr]

    assert not issues(with_group(ranking()))
    where = "smaa.groups[top]"
    cases = {
        "MISSING_ORDER": GroupUncertainty(parent_id=None, kind="ranking", held=["mandate"]),
        "INVALID_ORDER": ranking().model_copy(update={"order": ["team", "team", "market"]}),
        "UNEXPECTED_ORDER": GroupUncertainty(
            parent_id=None, kind="unknown", order=TOP, held=["mandate"]
        ),
        "UNEXPECTED_RATING_INTERVALS": ranking().model_copy(
            update={"intervals": {"team": (1.0, 2.0)}}
        ),
        "UNKNOWN_HELD_CHILD": ranking().model_copy(update={"held": ["mandate", "nope"]}),
        "DUPLICATE_HELD_CHILD": ranking().model_copy(update={"held": ["mandate", "mandate"]}),
    }
    for code, group in cases.items():
        found = {c for c, _ in issues(with_group(group))}
        assert code in found, (code, found)
    # a held child may not appear in the order, and the order must name every varying child
    bad = ranking().model_copy(update={"order": ["mandate", *TOP]})
    assert ("INVALID_ORDER", f"{where}.order") in issues(with_group(bad))
    short = ranking().model_copy(update={"order": TOP[:3]})
    assert ("INVALID_ORDER", f"{where}.order") in issues(with_group(short))
    # fewer than two varying children
    lonely = GroupUncertainty(
        parent_id=None, kind="unknown", held=["mandate", "team", "problem_solution", "market"]
    )
    assert ("INVALID_GROUP_UNCERTAINTY", f"{where}.held") in issues(with_group(lonely))
    # held children with all the weight: hold everything rated above 0 but two zero-rated ones
    zeroed = ps.set_judgement(ps.set_judgement(project, "market", 0.0, "technical_ip"), "traction",
                              0.0, "technical_ip")  # fmt: skip
    hold_all = GroupUncertainty(
        parent_id=None, kind="unknown", held=["mandate", "team", "problem_solution"]
    )
    model = SmaaModel(cells=project.smaa.cells, groups=[hold_all])
    assert ("HELD_TAKES_ALL_WEIGHT", f"{where}.held") in issues(ps.set_smaa_model(zeroed, model))
    # rating intervals for a held child are refused; held children need none
    intervals = {c: (40.0, 60.0) for c in TOP}
    ok = GroupUncertainty(parent_id=None, kind="rating_intervals", intervals=intervals,
                          held=["mandate"])  # fmt: skip
    assert not issues(with_group(ok))
    extra = ok.model_copy(update={"intervals": {**intervals, "mandate": (0.0, 0.0)}})
    assert ("UNEXPECTED_RATING_INTERVALS", f"{where}.intervals.mandate") in issues(
        with_group(extra)
    )
