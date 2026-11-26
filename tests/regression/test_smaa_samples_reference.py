"""SMAA-2 on stored weight samples agrees with SMAA-2 drawing its own weights.

- The analytic two-option case (order switches at a cost weight of 3/7): with uniform weights
  supplied as samples from an independent NumPy sampler, rank-1 acceptability is 4/7 and 3/7.
- Samples from Dirichlet(1, ..., 1) agree with the existing ``unknown`` path, on the tiny case and
  on case 2's top level with the mandate held.
- Samples from the conviction prior, Dirichlet(kappa * w0) with kappa 555 / 136 / 47 on case 2's
  top level, agree with an independent NumPy replica of SMAA-2 that draws Dirichlet weights itself.

Agreement between two sampled estimates is a two-proportion test at a family-wise alpha of 0.001
(Bonferroni over every compared frequency). The literal "each inside the other's 99.9% interval"
would fail about 2% of comparisons for correct code, because both sides carry noise. Against the
replica, which uses ten times the draws, the engine's interval at 1 - 0.001/m must contain the
replica's frequency.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.results import MavtResult, SmaaResult
from mcda_studio.domain.uncertainty import GroupUncertainty, SmaaModel
from mcda_studio.domain.weight_samples import WeightSamples
from mcda_studio.methods.smaa import samples as wsm
from mcda_studio.methods.smaa.intervals import clopper_pearson
from mcda_studio.services import analysis_service as svc
from mcda_studio.services import conviction_prior_service as cps
from mcda_studio.services import project_service as ps
from mcda_studio.storage.json_repository import JsonProjectRepository

from builders import tiny_project

pytestmark = pytest.mark.regression
ROOT = Path(__file__).resolve().parents[2]
CASE_2 = ROOT / "examples" / "nz_venture_screening" / "project.json"
FAMILY_ALPHA = 0.001


def sidecar_from_rows(
    project: Any, rows: np.ndarray, centre: list[float], kappa: float
) -> WeightSamples:
    """A prior-basis sidecar holding externally drawn global weight rows (leaf order)."""
    template = cps.conviction_prior(project, None, kappa=kappa, draws=1, seed=0)
    raw = json.loads(template.model_dump_json())
    raw["prior"]["groups"][0]["centre"] = centre
    raw["sampler"].update(draws=len(rows), samples=len(rows), algorithm="test: independent numpy")
    raw["samples"] = [[float(x) for x in row] for row in rows]
    return WeightSamples.model_validate(raw)


def smaa(project: Any, sidecar: WeightSamples | None, **parameters: Any) -> SmaaResult:
    params = dict(parameters)
    if sidecar is not None:
        params.update(weight_source="samples", weight_samples_sha256=wsm.digest(sidecar))
    cfg = AnalysisConfiguration(method="smaa2", weight_profile_id=project.weight_profiles[0].id,
                                parameters=params)  # fmt: skip
    with wsm.supplied(sidecar):
        result = svc.evaluate_project(project, cfg)
    assert isinstance(result, SmaaResult)
    return result


def agree(x: int, y: int, n: int, m: int) -> bool:
    """Two-proportion z-test of x/n against y/n at alpha FAMILY_ALPHA / m."""
    pooled = (x + y) / (2 * n)
    if pooled in (0.0, 1.0):
        return x == y
    z = (x - y) / n / math.sqrt(2 * pooled * (1 - pooled) / n)
    return 2 * stats.norm.sf(abs(z)) > FAMILY_ALPHA / m


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_uniform_samples_reproduce_four_sevenths(seed: int) -> None:
    project = tiny_project()
    rows = np.random.default_rng(1000 + seed).dirichlet([1.0, 1.0], size=20_000)
    sidecar = sidecar_from_rows(project, rows, [0.5, 0.5], 2.0)  # Dirichlet(2 * (1/2, 1/2))
    result = smaa(project, sidecar, uncertainty="preference", samples=20_000, seed=seed,
                  confidence=0.999)  # fmt: skip
    by = {row.alternative_id: row.ranks[0] for row in result.acceptability}
    assert by["a"].lower <= 4 / 7 <= by["a"].upper
    assert by["b"].lower <= 3 / 7 <= by["b"].upper


def compare(stated: SmaaResult, sampled: SmaaResult) -> None:
    pairs = []
    for a, b in zip(stated.acceptability, sampled.acceptability, strict=True):
        pairs.append((a.ranks[0].count, b.ranks[0].count))
        pairs += [(a.classes[c].count, b.classes[c].count) for c in a.classes]
    n = stated.samples
    assert n == sampled.samples and n is not None
    for x, y in pairs:
        assert agree(int(x), int(y), n, len(pairs)), (x, y, n)


def test_dirichlet_one_samples_agree_with_the_unknown_path_on_the_tiny_case() -> None:
    project = tiny_project()
    stated = smaa(ps.set_smaa_model(project, SmaaModel(groups=[GroupUncertainty(
        parent_id=None, kind="unknown")])), None, samples=40_000, seed=7)  # fmt: skip
    rows = np.random.default_rng(77).dirichlet([1.0, 1.0], size=40_000)
    sampled = smaa(project, sidecar_from_rows(project, rows, [0.5, 0.5], 2.0),
                   uncertainty="preference", samples=40_000, seed=7)  # fmt: skip
    compare(stated, sampled)


def case_2_top(project: Any) -> tuple[list[str], np.ndarray, list[int]]:
    """Leaves, each leaf's local weight within its objective, and each leaf's objective index."""
    tree = project.tree()
    local = svc.evaluate_project(
        project, AnalysisConfiguration(method="mavt", weight_profile_id="technical_ip")
    ).local_weights  # type: ignore[union-attr]
    top = [t for t in tree.child_ids(None) if t != "mandate"]
    leaves = tree.leaves()
    within = np.array([local[leaf] for leaf in leaves])
    parent = [top.index(tree.path(leaf)[0]) if tree.path(leaf)[0] in top else -1 for leaf in leaves]
    return leaves, within, parent


def global_rows(top_shares: np.ndarray, within: np.ndarray, parent: list[int]) -> np.ndarray:
    rows = np.zeros((len(top_shares), len(parent)))
    for j, k in enumerate(parent):
        if k >= 0:
            rows[:, j] = top_shares[:, k] * within[j]
    return rows


def test_dirichlet_one_samples_agree_with_the_unknown_path_on_case_2() -> None:
    project = JsonProjectRepository().load(CASE_2)
    cells = project.smaa.cells
    stated_project = ps.set_smaa_model(project, SmaaModel(cells=cells, groups=[GroupUncertainty(
        parent_id=None, kind="unknown", held=["mandate"])]))  # fmt: skip
    stated = smaa(stated_project, None, uncertainty="combined", samples=30_000, seed=11)
    plain = ps.set_smaa_model(project, SmaaModel(cells=cells))
    leaves, within, parent = case_2_top(plain)
    top = np.random.default_rng(111).dirichlet([1.0] * 4, size=30_000)
    sidecar = sidecar_from_rows(plain, global_rows(top, within, parent), [0.25] * 4, 4.0)
    assert sidecar.leaf_ids == leaves
    sampled = smaa(plain, sidecar, uncertainty="combined", samples=30_000, seed=11)
    compare(stated, sampled)


@pytest.fixture(scope="module")
def case_2_values() -> tuple[Any, list[str], np.ndarray]:
    project = JsonProjectRepository().load(CASE_2)
    fixed = project.model_copy(update={"smaa": None})  # entered measurements only
    mavt_cfg = AnalysisConfiguration(method="mavt", weight_profile_id="technical_ip")
    mavt = svc.evaluate_project(fixed, mavt_cfg)
    assert isinstance(mavt, MavtResult)
    order = [a.id for a in fixed.alternatives]
    by = {d.alternative_id: d for d in mavt.alternatives}
    leaves = mavt.leaf_order
    values = np.array([[by[a].partial_values[leaf] for leaf in leaves] for a in order])
    return fixed, leaves, values


@pytest.mark.parametrize("conviction", ["high", "medium", "low"])
def test_conviction_samples_agree_with_a_numpy_replica(case_2_values: Any, conviction: str) -> None:
    project, leaves, values = case_2_values
    sidecar = cps.conviction_prior(project, "technical_ip", conviction=conviction, draws=20_000,
                                   seed=21)  # fmt: skip
    result = smaa(project, sidecar, uncertainty="preference", samples=20_000, seed=21)
    group = next(g for g in sidecar.prior.groups if g.parent_id is None)
    assert group.kappa == {"high": 555, "medium": 136, "low": 47}[conviction]
    # The replica: its own Dirichlet draws for every varying group, stated local weights elsewhere,
    # global weights as products down each leaf's path.
    tree = project.tree()
    stated = svc.evaluate_project(
        project, AnalysisConfiguration(method="mavt", weight_profile_id="technical_ip")
    ).local_weights  # type: ignore[union-attr]
    rng = np.random.default_rng(2100)
    draws = 200_000
    local = {node: np.full(draws, stated[node]) for node in tree.all_ids()}
    for g in sidecar.prior.groups:
        shares = rng.dirichlet(g.kappa * np.asarray(g.centre), size=draws) * g.share
        for k, child in enumerate(g.children):
            local[child] = shares[:, k]
    rows = np.column_stack(
        [np.prod([local[n] for n in tree.path(leaf)], axis=0) for leaf in leaves]
    )
    totals = rows @ values.T  # (draws, alternatives)
    best = totals.max(axis=1, keepdims=True)
    first = np.isclose(totals, best, rtol=0, atol=1e-12)
    replica = first.mean(axis=0)
    m = len(result.acceptability)
    for k, row in enumerate(result.acceptability):
        count = row.ranks[0].count
        assert count is not None
        low, high = clopper_pearson(count, 20_000, 1 - FAMILY_ALPHA / m)
        assert low <= replica[k] <= high, (row.name, count / 20_000, replica[k])
