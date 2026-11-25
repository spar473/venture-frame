"""The conviction prior (``mcda calibrate prior``): exact Dirichlet(kappa * w0) draws of a profile's
weights, written as a ``weight-samples`` sidecar with ``basis: "prior"``.

Oracles: SciPy's Beta distribution for every Dirichlet component (KS), the exact means within six
standard errors, and the DD3 design note's table for kappa on case 2's top level.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from mcda_studio.domain import weight_samples as ws
from mcda_studio.domain.errors import McdaError
from mcda_studio.services import conviction_prior_service as cps
from mcda_studio.storage.json_repository import JsonProjectRepository

ROOT = Path(__file__).resolve().parents[2]
CASE_2 = ROOT / "examples" / "nz_venture_screening" / "project.json"


@pytest.fixture(scope="module")
def project() -> Any:
    return JsonProjectRepository().load(CASE_2)


@pytest.mark.parametrize(("conviction", "kappa"), [("high", 555), ("medium", 136), ("low", 47)])
def test_the_conviction_map_reproduces_555_136_47_on_case_2(project: Any, conviction: str,
                                                            kappa: int) -> None:  # fmt: skip
    samples = cps.conviction_prior(project, "technical_ip", conviction=conviction, draws=10)
    top = next(g for g in samples.prior.groups if g.parent_id is None)
    assert top.kappa == kappa
    assert top.children == ["team", "problem_solution", "market", "traction"]
    assert top.held == ["mandate"]
    assert top.kappa_method.startswith("Gauss-Legendre")
    assert samples.prior.conviction == conviction
    assert samples.prior.half_width == {"high": 0.15, "medium": 0.30, "low": 0.50}[conviction]


@pytest.fixture(scope="module")
def draws(project: Any) -> tuple[Any, np.ndarray]:
    samples = cps.conviction_prior(project, "technical_ip", conviction="low", draws=20_000, seed=7)
    return samples, np.array(samples.samples)


def test_every_component_has_its_exact_beta_marginal(project: Any, draws: Any) -> None:
    """KS against the exact Beta marginal for every component, at a family-wise alpha of 0.01
    (Bonferroni over the components): one test per component at 0.01 each would fail about one
    correct run in seven with 16 components. Means within six standard errors."""
    samples, w = draws
    tests = sum(len(g.children) for g in samples.prior.groups)
    leaves = samples.leaf_ids
    tree = project.tree()
    totals = {node: w[:, [leaves.index(x) for x in leaves if node in tree.path(x)]].sum(axis=1)
              for node in tree.all_ids()}  # fmt: skip
    for group in samples.prior.groups:
        for child, centre in zip(group.children, group.centre, strict=True):
            share = (
                totals[child]
                if group.parent_id is None
                else totals[child] / totals[group.parent_id]
            )
            free = share / group.share  # the component of Dirichlet(kappa * centre)
            a, b = group.kappa * centre, group.kappa * (1 - centre)
            p_value = stats.kstest(free, stats.beta(a, b).cdf).pvalue
            assert p_value > 0.01 / tests, (group.parent_id, child, p_value)
            se = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)) / len(free))
            assert abs(free.mean() - a / (a + b)) < 6 * se, (group.parent_id, child)


def test_held_and_zero_weight_children_stay_put(project: Any, draws: Any) -> None:
    samples, w = draws
    mandate = [
        i for i, leaf in enumerate(samples.leaf_ids) if leaf in ("hq_country", "stage", "sector")
    ]
    assert np.all(w[:, mandate] == 0.0)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert ("mandate", "zero-weight branch") in [
        (f.parent_id, f.reason) for f in samples.prior.fixed_groups
    ]


def test_global_means_are_the_stated_weights(project: Any, draws: Any) -> None:
    from mcda_studio.methods.mavt.weighting import compute_weights

    samples, w = draws
    stated = compute_weights(project.tree(), project.weight_profiles[0].judgements).global_
    for i, leaf in enumerate(samples.leaf_ids):
        sd = w[:, i].std()
        assert abs(w[:, i].mean() - stated[leaf]) < 6 * max(sd, 1e-12) / math.sqrt(len(w)), leaf
        assert samples.prior.centre_global[leaf] == pytest.approx(stated[leaf], abs=1e-12)


def test_a_stated_kappa_applies_to_every_varying_group(project: Any) -> None:
    samples = cps.conviction_prior(project, "commercial_traction", kappa=80.0, draws=5)
    assert {g.kappa for g in samples.prior.groups} == {80.0}
    assert {g.kappa_method for g in samples.prior.groups} == {"stated"}
    assert samples.prior.conviction is None and samples.prior.half_width is None
    with pytest.raises(McdaError, match="not both"):
        cps.conviction_prior(project, None, conviction="low", kappa=5.0, draws=5)
    with pytest.raises(McdaError, match="no conviction"):
        cps.conviction_prior(project, "technical_ip", draws=5)
    with pytest.raises(McdaError, match="draws must be"):
        cps.conviction_prior(project, "technical_ip", conviction="low", draws=0)


def test_draws_repeat_for_a_seed_and_never_change_the_project(project: Any) -> None:
    before = project.fingerprint()
    one = cps.conviction_prior(project, "technical_ip", conviction="medium", draws=50, seed=3)
    two = cps.conviction_prior(project, "technical_ip", conviction="medium", draws=50, seed=3)
    other = cps.conviction_prior(project, "technical_ip", conviction="medium", draws=50, seed=4)
    assert one.samples == two.samples and one.samples != other.samples
    assert cps.sidecar_digest(one) == cps.sidecar_digest(two)
    assert project.fingerprint() == before == one.project_fingerprint


def test_the_prior_sidecar_is_strict(project: Any) -> None:
    samples = cps.conviction_prior(project, "technical_ip", conviction="high", draws=20)
    raw = json.loads(samples.model_dump_json())
    assert raw["basis"] == "prior" and "reference_set_id" not in raw and "alpha" not in raw
    assert ws.WeightSamples.model_validate_json(samples.model_dump_json()) == samples

    def refused(change: Any, message: str) -> None:
        bad = json.loads(samples.model_dump_json())
        change(bad)
        with pytest.raises(ValidationError, match=message):
            ws.WeightSamples.model_validate_json(json.dumps(bad))

    refused(lambda r: r.update(reference_set_id="x"), "uses no reference set")
    refused(lambda r: r.update(alpha=[0.0] * 20), "uses no reference set")
    refused(lambda r: r["prior"].update(base_rate=0.15), "uses no reference set")
    refused(lambda r: r.update(review_states=["reviewed"]), "counts no ratings")
    refused(lambda r: r.update(usable=False), "is usable")
    refused(lambda r: r.update(label=ws.LABEL), "fixed prior label")
    refused(lambda r: r.update(basis="calibrated"), "a calibrated basis needs")
    refused(lambda r: r.update(basis="posterior"), "prior")


def test_the_command_writes_the_sidecar_and_prints_its_digest(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from mcda_studio.cli import app

    out = tmp_path / "prior.json"
    args = ["calibrate", "prior", str(CASE_2), "--profile", "technical_ip", "--conviction",
            "medium", "--draws", "200", "--seed", "1", "--out", str(out)]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    loaded = ws.WeightSamples.model_validate_json(out.read_text(encoding="utf-8"))
    assert summary["weight_samples_sha256"] == cps.sidecar_digest(loaded)
    method = loaded.prior.groups[0].kappa_method
    assert summary["groups"][0] == {"parent_id": None, "kappa": 136.0, "kappa_method": method}
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 1 and "Refusing to overwrite" in again.output
