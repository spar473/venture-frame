"""Robustness bands never invent uncertainty, never change the project and explain an absence."""

from pathlib import Path

from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.services import screening_robustness_service as svc
from mcda_studio.services import screening_summary_service as summary
from mcda_studio.storage.json_repository import JsonProjectRepository, to_json_text

EXAMPLES = Path(__file__).parents[2] / "examples"
repo = JsonProjectRepository()


def test_a_project_without_stated_uncertainty_is_unavailable_with_its_reason() -> None:
    project = repo.load(EXAMPLES / "supplier_qualification" / "project.json")
    bands = svc.screening_robustness(project)
    assert not bands.available
    assert bands.reason == "No stated uncertainty in this project."
    assert bands.rows == [] and bands.groups == [] and bands.draws is None


def test_a_run_with_nothing_uncertain_is_unavailable_with_the_engine_code() -> None:
    project = repo.load(EXAMPLES / "nz_venture_screening" / "project.json")
    assert project.smaa is not None
    cells_only = project.model_copy(
        update={
            "smaa": project.smaa.model_copy(update={"groups": []}),
            "analysis": AnalysisConfiguration(
                method="smaa2", parameters={"uncertainty": "preference"}
            ),
        }
    )
    bands = svc.screening_robustness(cells_only)
    assert not bands.available
    assert "NOTHING_UNCERTAIN" in bands.issue_codes
    assert bands.reason is not None and bands.reason.startswith("SMAA-2 cannot run: ")


def test_looking_changes_nothing_and_the_summary_is_untouched() -> None:
    project = repo.load(EXAMPLES / "nz_venture_screening" / "project.json")
    before, fingerprint = to_json_text(project), project.fingerprint()
    expected = summary.screening_summary(project)
    bands = svc.screening_robustness(project)
    assert to_json_text(project) == before and project.fingerprint() == fingerprint
    assert bands.project_fingerprint == fingerprint
    assert summary.screening_summary(project) == expected


def test_the_requested_configuration_is_echoed_and_transient_run_settings_are_ignored() -> None:
    project = repo.load(EXAMPLES / "nz_venture_screening" / "project.json")
    viewed = AnalysisConfiguration(
        method="smaa2", weight_profile_id="commercial_traction", parameters={"samples": 1000}
    )
    bands = svc.screening_robustness(project, viewed)
    assert bands.configuration == viewed
    assert bands.weight_profile_id == "commercial_traction"
    assert bands.draws == 10_000 and bands.parameters == "defaults"
