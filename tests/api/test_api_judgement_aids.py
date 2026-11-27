"""Judgement aids over HTTP: anchored swings, break-even weights and the bundled reference profile,
both routes equal to the service, read-only, and behind the cross-site guard."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import EXAMPLES, dump, import_file, repo
from fastapi.testclient import TestClient

from mcda_studio.api import create_app
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.services import judgement_aids_service as aids

STUDY = EXAMPLES / "nz_venture_screening"
BASE = "http://127.0.0.1:8765"


@pytest.fixture
def local(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(tmp_path / "projects", examples_dir=EXAMPLES, frontend_dir=False)
    with TestClient(app, base_url=BASE) as c:
        yield c


def test_draft_and_stored_routes_equal_the_service(local: TestClient) -> None:
    project = repo.load(STUDY / "project.json")
    configuration = {"method": "mavt_sorting", "weight_profile_id": "technical_ip"}
    references = aids.load_reference_profiles(EXAMPLES / "reference_profiles")
    assert [r.id for r in references] == ["vc_literature"]
    expected = aids.judgement_aids(project, AnalysisConfiguration(**configuration), references)
    draft = local.post(
        "/api/judgement-aids", json={"project": dump(project), "configuration": configuration}
    )
    assert draft.status_code == 200, draft.text
    body = draft.json()
    assert body == expected.model_dump(mode="json")
    assert body["available"] is True and body["break_even"]["leader_name"] == "Upstock"
    assert any(
        s.startswith("Upstock stays first unless Team falls below")
        for s in (c["sentence"] for c in body["break_even"]["criteria"])
    )
    anchor = next(a for a in body["anchors"] if a["criterion_id"] == "team_completeness")
    assert anchor["swing"].startswith("From ") and " to " in anchor["swing"]
    (reference,) = body["references"]
    assert reference["partner_order"] == ["problem_solution", "team", "market"]
    variants = {v["id"]: v for v in reference["variants"]}
    assert variants["stated_by_vcs"]["disagreements"] == 1  # Problem and solution above Team
    assert variants["outcome_predictive"]["disagreements"] == 2
    stored_id = import_file(local, STUDY / "project.json")["project"]["id"]
    before = local.get(f"/api/projects/{stored_id}").json()
    stored = local.post(
        f"/api/projects/{stored_id}/judgement-aids", json={"configuration": configuration}
    )
    assert stored.status_code == 200 and stored.json() == body
    assert local.get(f"/api/projects/{stored_id}").json() == before


def test_without_the_examples_folder_there_is_no_reference(tmp_path: Path) -> None:
    (tmp_path / "examples").mkdir()  # an examples folder without reference profiles
    app = create_app(tmp_path / "projects", examples_dir=tmp_path / "examples", frontend_dir=False)
    with TestClient(app, base_url=BASE) as c:
        project = repo.load(STUDY / "project.json")
        body = c.post("/api/judgement-aids", json={"project": dump(project)}).json()
    assert body["available"] is True and body["references"] == []


def test_an_incomplete_model_says_why(local: TestClient) -> None:
    project = repo.load(STUDY / "project.json")
    raw = dump(project)
    raw["performances"] = raw["performances"][1:]
    body = local.post("/api/judgement-aids", json={"project": raw}).json()
    assert body["available"] is False and body["reason"].startswith("the model cannot be evaluated")


def test_a_reference_never_becomes_a_default(local: TestClient) -> None:
    project = repo.load(STUDY / "project.json")
    before = dump(project)
    local.post("/api/judgement-aids", json={"project": before})
    assert dump(project) == before
    assert all("reference" not in p["id"] for p in before["weight_profiles"])


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://example.com", "Sec-Fetch-Site": "cross-site"},
        {"Origin": BASE, "Sec-Fetch-Site": "cross-site"},
    ],
)
def test_a_cross_site_request_is_refused(local: TestClient, headers: dict[str, str]) -> None:
    project = repo.load(STUDY / "project.json")
    for path in ("/api/judgement-aids", "/api/projects/x/judgement-aids"):
        body = {"project": dump(project)} if "projects" not in path else {}
        response = local.post(path, json=body, headers=headers)
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "CROSS_ORIGIN_REFUSED"
