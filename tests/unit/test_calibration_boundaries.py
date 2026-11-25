"""Calibration stays domain-neutral and offline: no domain words in its code or fixed text, and
neither the API nor the interface can reach it (only the command line imports it).
"""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "mcda_studio" / "calibration"
DOMAIN_WORDS = (
    "venture", "startup", "start-up", "funding", "investor", "investment", "founder", "company",
    "companies", "series a", "seed round", "valuation", "portfolio", "unicorn", "exit",
)  # fmt: skip


def test_the_calibration_package_uses_no_domain_words() -> None:
    files = sorted(PACKAGE.glob("*.py"))
    assert {f.name for f in files} >= {
        "prior.py", "likelihood.py", "sampler.py", "diagnostics.py", "synthetic.py",
        "weight_samples.py", "service.py",
    }  # fmt: skip
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for word in DOMAIN_WORDS:
            assert word not in text, f"{path.name} mentions {word!r}"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_only_the_command_line_imports_calibration() -> None:
    source = ROOT / "src" / "mcda_studio"
    for path in source.rglob("*.py"):
        if PACKAGE in path.parents or path.name == "cli.py":
            continue
        assert not any(n.startswith("mcda_studio.calibration") for n in _imports(path)), path


def test_the_api_never_loads_calibration() -> None:
    code = (
        "import sys; from mcda_studio.api import create_app; create_app(); "
        "print(sorted(m for m in sys.modules if m.startswith('mcda_studio.calibration')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == "[]"


def test_the_interface_never_mentions_calibration() -> None:
    for path in (ROOT / "frontend" / "src").rglob("*"):
        if path.is_file() and path.suffix in {".ts", ".tsx", ".js", ".jsx", ".json", ".css"}:
            assert "calibrat" not in path.read_text(encoding="utf-8").lower(), path
    openapi = (ROOT / "docs" / "api" / "openapi.json").read_text(encoding="utf-8").lower()
    assert "calibrat" not in openapi and "weight-samples" not in openapi
