"""The cross-site guard on a real `mcda serve`: the serving port comes from uvicorn's socket."""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post(url: str, headers: dict[str, str]) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"title": "Origin check"}).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_a_real_server_refuses_other_origins_and_accepts_its_own(tmp_path):
    port = free_port()
    command = [sys.executable, "-m", "mcda_studio", "serve", "--store", str(tmp_path / "library")]
    command += ["--port", str(port), "--no-open"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            assert process.poll() is None, process.communicate()[0]
            try:
                urllib.request.urlopen(base + "/api/health", timeout=5).close()
                break
            except OSError:
                time.sleep(0.025)
        url = base + "/api/projects"
        status, body = post(url, {"Origin": f"http://127.0.0.1:{port + 1}"})
        assert status == 403 and body["error"]["code"] == "CROSS_ORIGIN_REFUSED"
        status, body = post(url, {"Origin": "https://example.com"})
        assert status == 403 and body["error"]["code"] == "CROSS_ORIGIN_REFUSED"
        status, body = post(url, {"Sec-Fetch-Site": "cross-site"})
        assert status == 403 and body["error"]["code"] == "CROSS_ORIGIN_REFUSED"
        status, _ = post(
            url, {"Origin": f"http://localhost:{port}", "Sec-Fetch-Site": "same-origin"}
        )
        assert status == 201
        status, _ = post(url, {})
        assert status == 201
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
