"""The explicit cross-site guard: Origin and Sec-Fetch-Site on state-changing requests.

The test client serves on the base URL's host and port, as uvicorn serves on its socket's, so the
"serving port" here is 8765. A request with no browser headers is how the CLI, curl and the other
API tests talk to the server, and stays allowed.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import EXAMPLES
from fastapi.testclient import TestClient

from mcda_studio.api import create_app
from mcda_studio.api.origin import refusal

BASE = "http://127.0.0.1:8765"
NEW = {"title": "Origin check"}


@pytest.fixture
def local(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(tmp_path / "projects", examples_dir=EXAMPLES, frontend_dir=False)
    with TestClient(app, base_url=BASE) as c:
        yield c


def refused(response, reason: str) -> None:
    assert response.status_code == 403, response.text
    error = response.json()["error"]
    assert error["code"] == "CROSS_ORIGIN_REFUSED"
    assert "another website" in error["message"]
    assert reason in error["details"][0]["message"]


@pytest.mark.parametrize(
    "origin",
    [
        "https://example.com",
        "http://evil.localhost.example:8765",
        "null",
        "https://127.0.0.1:8765",  # wrong scheme for this server
        "http://192.168.1.10:8765",
    ],
)
def test_a_cross_site_origin_post_is_refused(local, origin):
    response = local.post("/api/projects", json=NEW, headers={"Origin": origin})
    refused(response, "Origin")
    assert local.get("/api/projects").json()["projects"] == []


@pytest.mark.parametrize("site", ["cross-site", "same-site", "CROSS-SITE"])
def test_a_cross_site_fetch_is_refused_even_with_a_local_origin(local, site):
    headers = {"Sec-Fetch-Site": site, "Origin": BASE}
    refused(local.post("/api/projects", json=NEW, headers=headers), "Sec-Fetch-Site")
    assert local.get("/api/projects").json()["projects"] == []


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:5173", "http://localhost:8766", "http://[::1]"]
)
def test_a_local_origin_on_another_port_is_refused(local, origin):
    refused(local.post("/api/projects", json=NEW, headers={"Origin": origin}), "Origin")


@pytest.mark.parametrize("origin", [BASE, "http://localhost:8765", "http://[::1]:8765"])
def test_a_same_origin_browser_post_is_allowed(local, origin):
    headers = {"Origin": origin, "Sec-Fetch-Site": "same-origin"}
    response = local.post("/api/projects", json=NEW, headers=headers)
    assert response.status_code == 201, response.text


def test_a_post_without_browser_headers_is_allowed(local):
    assert local.post("/api/projects", json=NEW).status_code == 201
    assert local.post(
        "/api/projects", json=NEW, headers={"Sec-Fetch-Site": "none"}
    ).status_code == (201)


def test_every_state_changing_method_is_guarded(local):
    made = local.post("/api/projects", json=NEW).json()
    path = f"/api/projects/{made['project']['id']}"
    foreign = {"Origin": "https://example.com"}
    put = local.put(
        path, json=made["project"], headers={**foreign, "If-Match": made["fingerprint"]}
    )
    refused(put, "Origin")
    refused(local.delete(path, headers=foreign), "Origin")
    refused(local.request("PATCH", path, json={}, headers=foreign), "Origin")
    assert local.get(path).json()["fingerprint"] == made["fingerprint"]


def test_safe_methods_are_not_affected_by_a_foreign_origin(local):
    foreign = {"Origin": "https://example.com", "Sec-Fetch-Site": "cross-site"}
    assert local.get("/api/health", headers=foreign).status_code == 200
    assert local.get("/api/projects", headers=foreign).status_code == 200
    assert local.head("/api/health", headers=foreign).status_code != 403
    assert local.options("/api/health", headers=foreign).status_code != 403


def test_the_default_test_client_without_headers_still_works(client):
    """The shared fixture (base URL http://testserver) sends no browser headers."""
    assert client.post("/api/projects", json=NEW).status_code == 201


def test_refusal_reads_the_serving_port_from_the_host_header_without_a_server_address():
    scope = {
        "type": "http",
        "method": "POST",
        "server": None,
        "headers": [(b"host", b"localhost:9000"), (b"origin", b"http://localhost:9000")],
    }
    assert refusal(scope) is None
    scope["headers"] = [(b"host", b"localhost:9000"), (b"origin", b"http://localhost:9001")]
    assert refusal(scope) is not None
