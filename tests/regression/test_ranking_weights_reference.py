"""The ranking weight kind and held children, against theory and an independent sampler.

A ranking group's local weights should be uniform on the part of the simplex that keeps the stated
order. Three independent facts pin that down:

1. every draw keeps the order (exact);
2. the mean draw is the rank-order centroid (ROC) of Barron and Barrett (1996), which is the
   centroid of the ordered region (theory, checked to six standard errors);
3. for two children the larger weight is uniform on [1/2, 1] (exact distribution, KS test), and
   for three children each coordinate matches a rejection sampler built from NumPy's own
   Dirichlet draws, a code path the engine does not use (two-sample KS tests).

Held children keep the profile's local weight in every draw. On case 2, SMAA-2 with a ranking
over the four objectives and the mandate held matches a NumPy replica of the same model within
SMAA-2's exact 99.9% intervals. All seeds are fixed, so each run is deterministic; the statistical
thresholds (six standard errors, KS at alpha = 0.001) say how surprising a failure would be.
Design: ``docs/design/structured-judgement.md``.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import ks_2samp, kstest

from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.results import MavtResult, SmaaResult
from mcda_studio.domain.uncertainty import GroupUncertainty, SmaaModel
from mcda_studio.methods.smaa import engine
from mcda_studio.services import analysis_service
from mcda_studio.services import elicitation_service as es
from mcda_studio.services import project_service as ps
from mcda_studio.storage.json_repository import JsonProjectRepository

from builders import flat_project

STUDY = Path(__file__).parents[2] / "examples" / "nz_venture_screening" / "project.json"
PREFERENCE = AnalysisConfiguration(method="smaa2", parameters={"uncertainty": "preference"})
N = 200_000
ALPHA = 0.001


def local_draws(project, n: int = N, seed: int = 11) -> dict[str, np.ndarray]:  # type: ignore[no-untyped-def]
    """The engine's own local weights for the project's groups, from its own uniform draws."""
    plan = engine._plan(project, PREFERENCE, "preference")
    width = sum(len(g.varying) for g in plan.groups)
    uniforms = np.random.Generator(np.random.PCG64(seed)).random((n, width))
    local, used = engine._group_locals(plan, uniforms)
    assert used == width
    return local


def flat_with_group(ratings: list[float], group: GroupUncertainty):  # type: ignore[no-untyped-def]
    values = [[0.1 * (i + j) % 1 for j in range(len(ratings))] for i in range(3)]
    project = flat_project(values, ratings)
    return ps.set_smaa_model(project, SmaaModel(groups=[group]))


def test_every_draw_keeps_the_order_and_the_mean_is_the_rank_order_centroid() -> None:
    order = ["c2", "c0", "c3", "c1"]
    project = flat_with_group([10.0, 20.0, 30.0, 40.0], GroupUncertainty(
        parent_id=None, kind="ranking", order=order))  # fmt: skip
    local = local_draws(project)
    w = np.column_stack([local[c] for c in order])
    assert np.all(np.diff(w, axis=1) <= 0.0)  # (1) exact
    assert np.allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    roc = es.roc_weights(4)
    assert roc == pytest.approx([25 / 48, 13 / 48, 7 / 48, 3 / 48], abs=1e-15)
    se = w.std(axis=0, ddof=1) / math.sqrt(N)
    assert np.all(np.abs(w.mean(axis=0) - roc) < 6 * se), (w.mean(axis=0), roc, se)  # (2)


def test_two_children_the_larger_weight_is_uniform_on_a_half_to_one() -> None:
    project = flat_with_group([1.0, 1.0], GroupUncertainty(
        parent_id=None, kind="ranking", order=["c1", "c0"]))  # fmt: skip
    first = local_draws(project)["c1"]
    assert first.min() >= 0.5 and first.max() <= 1.0
    assert kstest(first, "uniform", args=(0.5, 0.5)).pvalue > ALPHA  # (3) exact distribution


def test_three_children_match_an_independent_rejection_sampler() -> None:
    order = ["c0", "c1", "c2"]
    project = flat_with_group([1.0, 1.0, 1.0], GroupUncertainty(
        parent_id=None, kind="ranking", order=order))  # fmt: skip
    local = local_draws(project, n=60_000)
    engine_w = np.column_stack([local[c] for c in order])
    rng = np.random.default_rng(2026)
    d = rng.dirichlet(np.ones(3), 400_000)
    oracle = d[(d[:, 0] >= d[:, 1]) & (d[:, 1] >= d[:, 2])]  # uniform, kept only if ordered
    assert len(oracle) > 60_000
    for k in range(3):
        assert ks_2samp(engine_w[:, k], oracle[:, k]).pvalue > ALPHA, k
    product = ks_2samp(engine_w[:, 0] * engine_w[:, 2], oracle[:, 0] * oracle[:, 2])
    assert product.pvalue > ALPHA  # a joint feature, not only the margins


@pytest.mark.parametrize(
    "group",
    [
        GroupUncertainty(parent_id=None, kind="unknown", held=["c0"]),
        GroupUncertainty(parent_id=None, kind="ranking", order=["c3", "c1", "c2"], held=["c0"]),
        GroupUncertainty(
            parent_id=None,
            kind="rating_intervals",
            intervals={"c1": (20.0, 40.0), "c2": (10.0, 30.0), "c3": (5.0, 15.0)},
            held=["c0"],
        ),
    ],
)
def test_held_children_keep_the_profile_weight_and_the_rest_share_what_is_left(
    group: GroupUncertainty,
) -> None:
    project = flat_with_group([40.0, 30.0, 20.0, 10.0], group)  # c0's local weight is 0.4
    local = local_draws(project, n=20_000)
    assert np.all(local["c0"] == 0.4)
    rest = local["c1"] + local["c2"] + local["c3"]
    assert np.allclose(rest, 0.6, rtol=0, atol=1e-12)
    assert np.all(local["c1"] >= 0) and local["c1"].std() > 0  # the others vary


def test_a_zero_rated_held_child_stays_at_zero() -> None:
    project = flat_with_group([0.0, 30.0, 20.0], GroupUncertainty(
        parent_id=None, kind="unknown", held=["c0"]))  # fmt: skip
    local = local_draws(project, n=5_000)
    assert np.all(local["c0"] == 0.0)
    assert np.allclose(local["c1"] + local["c2"], 1.0, rtol=0, atol=1e-12)


def _case2_ranking():  # type: ignore[no-untyped-def]
    project = JsonProjectRepository().load(STUDY)
    group = GroupUncertainty(
        parent_id=None,
        kind="ranking",
        order=["problem_solution", "team", "market", "traction"],
        held=["mandate"],
    )
    assert project.smaa is not None
    return ps.set_smaa_model(project, SmaaModel(cells=project.smaa.cells, groups=[group]))


def test_case_2_ranking_matches_a_numpy_replica_within_exact_intervals() -> None:
    project = _case2_ranking()
    configuration = PREFERENCE.model_copy(update={"weight_profile_id": "technical_ip"})
    smaa = analysis_service.evaluate_project(project, configuration)
    assert isinstance(smaa, SmaaResult) and smaa.samples == 10_000
    mavt = analysis_service.evaluate_project(
        project, configuration.model_copy(update={"method": "mavt", "parameters": {}})
    )
    assert isinstance(mavt, MavtResult)
    objectives = ["problem_solution", "team", "market", "traction"]
    parent = {n.id: n.parent_id for n in project.criteria}
    detail = {d.alternative_id: d for d in mavt.alternatives}
    ids = [a.id for a in project.alternatives]

    def score(a: str, g: str) -> float:
        leaves = [leaf for leaf in mavt.leaf_order if parent[leaf] == g]
        return math.fsum(mavt.local_weights[x] * detail[a].partial_values[x] for x in leaves)

    scores = np.array([[score(a, g) for g in objectives] for a in ids])
    rng = np.random.default_rng(7)
    weights = -np.sort(-rng.dirichlet(np.ones(4), 400_000), axis=1)  # ordered, best first
    values = weights @ scores.T
    first = (values == values.max(axis=1, keepdims=True)).mean(axis=0)
    prioritise = (values >= 0.6).mean(axis=0)
    by = {a.alternative_id: a for a in smaa.acceptability}
    n = smaa.samples
    for k, a in enumerate(ids):
        for estimate, replica in [
            (by[a].ranks[0], first[k]),
            (by[a].classes["prioritise_for_diligence"], prioritise[k]),
        ]:
            count = estimate.count
            assert count is not None
            lo = 0.0 if count == 0 else beta_dist.ppf(ALPHA / 2, count, n - count + 1)
            hi = 1.0 if count == n else beta_dist.ppf(1 - ALPHA / 2, count + 1, n - count)
            assert lo <= replica <= hi, (a, estimate.frequency, replica)


def test_held_mandate_changes_no_other_weight_than_the_varying_ones() -> None:
    project = _case2_ranking()
    configuration = PREFERENCE.model_copy(update={"weight_profile_id": "technical_ip"})
    local = engine._group_locals(
        engine._plan(project, configuration, "preference"),
        np.random.Generator(np.random.PCG64(0)).random((1000, 4)),
    )[0]
    assert np.all(local["mandate"] == 0.0)
    assert set(local) == {"mandate", "problem_solution", "team", "market", "traction"}


def test_a_ranked_child_is_an_input_judged_by_its_local_weight() -> None:
    project = _case2_ranking()
    configuration = PREFERENCE.model_copy(update={"weight_profile_id": "technical_ip"})
    smaa = analysis_service.evaluate_project(project, configuration)
    assert isinstance(smaa, SmaaResult)
    labels = [i.input for i in smaa.inputs_to_check]
    assert "local weight of Team under the top level" in labels
    assert not any("swing rating" in label or "Mandate" in label for label in labels)
