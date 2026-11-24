"""Robustness bands over HTTP: both routes return the service's own result, write nothing, and sit
behind the cross-site guard like every other POST route."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import EXAMPLES, dump, import_file, repo
from fastapi.testclient import TestClient

from mcda_studio.api import create_app
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.services import screening_robustness_service as robustness

STUDY = EXAMPLES / "nz_venture_screening"
BASE = "http://127.0.0.1:8765"


@pytest.fixture
def local(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(tmp_path / "projects", examples_dir=EXAMPLES, frontend_dir=False)
    with TestClient(app, base_url=BASE) as c:
        yield c


def test_draft_and_stored_routes_equal_the_service(client: TestClient) -> None:
    project = repo.load(STUDY / "project.json")
    configuration = {"method": "mavt_sorting", "weight_profile_id": "technical_ip"}
    expected = robustness.screening_robustness(project, AnalysisConfiguration(**configuration))
    draft = client.post(
        "/api/screening-summary/robustness",
        json={"project": dump(project), "configuration": configuration},
    )
    assert draft.status_code == 200, draft.text
    assert draft.json() == expected.model_dump(mode="json")
    assert draft.json()["available"] is True
    stored_id = import_file(client, STUDY / "project.json")["project"]["id"]
    before = client.get(f"/api/projects/{stored_id}").json()
    stored = client.post(
        f"/api/projects/{stored_id}/screening-summary/robustness",
        json={"configuration": configuration},
    )
    assert stored.status_code == 200 and stored.json() == draft.json()
    assert client.get(f"/api/projects/{stored_id}").json() == before


def test_a_project_without_stated_uncertainty_is_unavailable(client: TestClient) -> None:
    project = repo.load(EXAMPLES / "supplier_qualification" / "project.json")
    body = client.post("/api/screening-summary/robustness", json={"project": dump(project)}).json()
    assert body["available"] is False
    assert body["reason"] == "No stated uncertainty in this project."


def test_a_same_origin_browser_request_is_served(local: TestClient) -> None:
    project = repo.load(STUDY / "project.json")
    headers = {"Origin": BASE, "Sec-Fetch-Site": "same-origin"}
    response = local.post(
        "/api/screening-summary/robustness", json={"project": dump(project)}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.json()["available"] is True


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://example.com", "Sec-Fetch-Site": "cross-site"},
        {"Origin": BASE, "Sec-Fetch-Site": "cross-site"},
    ],
)
def test_a_cross_site_request_is_refused(local: TestClient, headers: dict[str, str]) -> None:
    project = repo.load(STUDY / "project.json")
    for path in (
        "/api/screening-summary/robustness",
        "/api/projects/x/screening-summary/robustness",
    ):
        body = {"project": dump(project)} if "projects" not in path else {}
        response = local.post(path, json=body, headers=headers)
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "CROSS_ORIGIN_REFUSED"
