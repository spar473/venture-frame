"""Structured judgement over HTTP and the CLI: the routes return the service's own results and
write nothing; the edit operation records and clears; the CLI states ranking groups and held
children and prints the check."""

import json
from pathlib import Path

import pytest
from conftest import EXAMPLES, dump, import_file, repo
from fastapi.testclient import TestClient
from rich.console import Console
from typer.testing import CliRunner

from mcda_studio import cli
from mcda_studio.cli import app as cli_app
from mcda_studio.domain.preferences import AnalysisConfiguration, Elicitation, GroupElicitation
from mcda_studio.services import elicitation_service as es
from mcda_studio.services import project_service as ps

STUDY = EXAMPLES / "nz_venture_screening" / "project.json"
ORDER = ["problem_solution", "team", "market", "traction"]
RECORD = Elicitation(
    elicited_by="Study author",
    conviction="low",
    groups=[GroupElicitation(parent_id=None, order=ORDER, reasons={"team": "verifiable"})],
)


def test_check_and_preview_routes_equal_the_service(client: TestClient) -> None:
    project = ps.set_elicitation(repo.load(STUDY), "technical_ip", RECORD)
    configuration = {"method": "mavt_sorting", "weight_profile_id": "technical_ip"}
    body = {"project": dump(project), "configuration": configuration}
    check = client.post("/api/elicitation/check", json=body)
    assert check.status_code == 200, check.text
    assert check.json() == es.elicitation_check(project, "technical_ip").model_dump(mode="json")
    assert [p["mode"] for p in check.json()["proposals"]] == ["conviction", "ranking"]
    preview = client.post("/api/elicitation/preview", json=body)
    assert preview.status_code == 200, preview.text
    expected = es.elicitation_preview(project, AnalysisConfiguration(**configuration))
    assert preview.json() == expected.model_dump(mode="json")


def test_the_edit_operation_records_and_stored_routes_read_without_writing(
    client: TestClient,
) -> None:
    project = repo.load(STUDY)
    edited = client.post(
        "/api/edit",
        json={
            "project": dump(project),
            "operations": [
                {
                    "op": "set_elicitation",
                    "profile_id": "commercial_traction",
                    "elicitation": RECORD.model_dump(mode="json"),
                }
            ],
        },
    )
    assert edited.status_code == 200, edited.text
    document = edited.json()["project"]
    assert document["schema_version"] == 7
    profile = next(p for p in document["weight_profiles"] if p["id"] == "commercial_traction")
    assert profile["elicitation"]["conviction"] == "low"
    stored_id = import_file(client, STUDY)["project"]["id"]
    before = client.get(f"/api/projects/{stored_id}").json()
    for path in ("check", "preview"):
        response = client.post(
            f"/api/projects/{stored_id}/elicitation/{path}",
            json={"configuration": {"method": "mavt", "weight_profile_id": "technical_ip"}},
        )
        assert response.status_code == 200, response.text
    assert client.get(f"/api/projects/{stored_id}").json() == before


def test_an_unknown_profile_is_a_structured_error(client: TestClient) -> None:
    project = repo.load(STUDY)
    response = client.post(
        "/api/elicitation/check",
        json={"project": dump(project), "configuration": {"weight_profile_id": "nobody"}},
    )
    assert response.status_code in {400, 404, 422}, response.text
    assert response.json()["error"]["code"]


def test_the_cli_states_a_ranking_group_with_a_held_child_and_prints_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=160))
    monkeypatch.setattr(cli, "err_console", Console(stderr=True, width=160))
    path = tmp_path / "p.json"
    path.write_bytes(STUDY.read_bytes())
    runner = CliRunner()

    def run(*args: str, code: int = 0) -> str:
        result = runner.invoke(cli_app, list(args))
        assert result.exit_code == code, result.output
        return result.output

    args = ["uncertainty", "group", str(path)]
    for child in ORDER:
        args += ["--rank", child]
    run(*args, "--hold", "mandate")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 7
    (group,) = [g for g in raw["smaa"]["groups"] if g["parent_id"] is None]
    assert group == {"parent_id": None, "kind": "ranking", "intervals": {}, "order": ORDER,
                     "held": ["mandate"]}  # fmt: skip
    out = run("evaluate", str(path), "--method", "smaa2", "--uncertainty", "preference",
              "--samples", "1000")  # fmt: skip
    assert "(preference run; 1000 draws" in out
    assert "give exactly one of" in run(*args, "--unknown", code=1)
    printed = json.loads(run("elicitation-check", str(path), "--profile", "technical_ip"))
    assert printed["profile_id"] == "technical_ip" and printed["recorded"] is False
