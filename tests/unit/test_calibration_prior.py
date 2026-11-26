"""Calibration layer 1: the ALR transform and its Jacobian, the nested Dirichlet prior's structure
(held children, zero-weight branches, centres), its exact moments and the moment-matched kappa.

Oracles: finite differences for the Jacobian, SciPy's Dirichlet density, Monte Carlo moments of
NumPy's own Dirichlet draws, and the DD3 design note's table for kappa on case 2's top level.
"""

import math

import numpy as np
import pytest
from scipy import stats

from mcda_studio.calibration import prior as pr
from mcda_studio.calibration import synthetic as sy

RNG = np.random.Generator(np.random.PCG64(20261125))


# ---- the transform ----------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_alr_round_trips_both_ways(k: int) -> None:
    p = RNG.dirichlet(np.ones(k), size=200)
    assert np.allclose(pr.alr_inverse(pr.alr(p)), p, rtol=0, atol=1e-13)
    y = RNG.normal(0.0, 3.0, size=(200, k - 1))
    assert np.allclose(pr.alr(pr.alr_inverse(y)), y, rtol=0, atol=1e-10)
    assert np.allclose(pr.alr_inverse(y).sum(axis=1), 1.0, atol=1e-15)


def test_alr_inverse_is_stable_for_extreme_coordinates() -> None:
    y = np.array([[800.0, -800.0], [-800.0, -800.0]])
    p = pr.alr_inverse(y)
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1.0)
    assert np.all(np.isfinite(pr.log_simplex(y)))


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_log_jacobian_matches_finite_differences(k: int) -> None:
    for _ in range(10):
        y = RNG.normal(0.0, 1.5, size=k - 1)
        h = 1e-6
        jac = np.empty((k - 1, k - 1))
        for j in range(k - 1):
            up, down = y.copy(), y.copy()
            up[j] += h
            down[j] -= h
            jac[:, j] = (pr.alr_inverse(up)[:-1] - pr.alr_inverse(down)[:-1]) / (2 * h)
        sign, logdet = np.linalg.slogdet(jac)
        assert sign > 0
        assert pr.log_jacobian(y) == pytest.approx(logdet, abs=1e-7)


def test_group_density_is_the_dirichlet_density_plus_the_jacobian() -> None:
    centre, kappa = (0.5, 0.3, 0.2), 12.0
    for _ in range(20):
        y = RNG.normal(0.0, 1.0, size=2)
        p = pr.alr_inverse(y)
        expected = stats.dirichlet.logpdf(p, kappa * np.array(centre)) + pr.log_jacobian(y)
        assert pr.group_log_density(y, centre, kappa) == pytest.approx(expected, abs=1e-9)


def test_group_density_integrates_to_one_in_alr_coordinates() -> None:
    # Two children: y is one-dimensional, so a fine grid integrates the density directly.
    grid = np.linspace(-40.0, 40.0, 400_001)
    density = np.exp(pr.group_log_density(grid[:, None], (0.7, 0.3), 5.0))
    assert float(density.sum() * (grid[1] - grid[0])) == pytest.approx(1.0, abs=1e-6)


# ---- kappa from a conviction -------------------------------------------------------------------


@pytest.mark.parametrize(("conviction", "half_width", "kappa"), [
    ("high", 0.15, 555), ("medium", 0.30, 136), ("low", 0.50, 47),
])  # fmt: skip
def test_kappa_on_case_2s_top_level_matches_the_design_note(
    conviction: str, half_width: float, kappa: int
) -> None:
    from mcda_studio.services.elicitation_service import CONVICTION_HALF_WIDTH

    assert CONVICTION_HALF_WIDTH[conviction] == half_width
    result = pr.moment_matched_kappa([90, 100, 60, 50], half_width)
    assert result.kappa == kappa
    assert result.method.startswith("Gauss-Legendre")


def test_kappa_agrees_with_a_monte_carlo_of_the_rating_intervals() -> None:
    ratings, h = np.array([30.0, 20.0, 50.0]), 0.3
    x = RNG.uniform(ratings * (1 - h), ratings * (1 + h), size=(2_000_000, 3))
    q = x / x.sum(axis=1, keepdims=True)
    w = ratings / ratings.sum()
    monte_carlo = float(np.mean(w * (1 - w) / q.var(axis=0))) - 1
    assert pr.moment_matched_kappa(ratings, h).kappa == pytest.approx(monte_carlo, rel=0.01)


def test_kappa_falls_back_to_the_delta_method_for_large_groups_and_stays_close() -> None:
    ratings = list(range(1, 14))  # 13 free children
    result = pr.moment_matched_kappa(ratings, 0.3)
    assert result.method == "first-order delta method"
    r = np.array(ratings, dtype=float)
    x = RNG.uniform(r * 0.7, r * 1.3, size=(400_000, r.size))
    q = x / x.sum(axis=1, keepdims=True)
    w = r / r.sum()
    assert result.kappa == pytest.approx(float(np.mean(w * (1 - w) / q.var(axis=0))) - 1, rel=0.05)


def test_wider_conviction_bands_mean_smaller_kappa() -> None:
    ks = [pr.moment_matched_kappa([3, 2, 1], h).kappa for h in (0.15, 0.30, 0.50)]
    assert ks[0] > ks[1] > ks[2] >= 1


# ---- the nested prior ---------------------------------------------------------------------------

NODES = ("o1", "a", "b", "c", "o2", "d", "e", "o3", "f", "g")
CHILDREN = {None: ("o1", "o2", "o3"), "o1": ("a", "b", "c"), "o2": ("d", "e"), "o3": ("f", "g")}
LOCAL = {"o1": 0.5, "o2": 0.5, "o3": 0.0, "a": 0.6, "b": 0.4, "c": 0.0, "d": 0.7, "e": 0.3,
         "f": 0.5, "g": 0.5}  # fmt: skip


def test_rated_zero_children_are_held_and_zero_weight_branches_are_fixed() -> None:
    prior = pr.make_prior(NODES, CHILDREN, LOCAL, {None: 10.0, "o1": 20.0, "o2": 30.0})
    groups = {g.parent_id: g for g in prior.groups}
    assert set(groups) == {None, "o1", "o2"}
    assert groups[None].children == ("o1", "o2") and groups[None].held == ("o3",)
    assert groups["o1"].children == ("a", "b") and groups["o1"].held == ("c",)
    assert [(f.parent_id, f.reason) for f in prior.fixed] == [("o3", "zero-weight branch")]
    ys = prior.sample_ys(RNG, 500)
    w = prior.global_weights(ys)
    leaves = list(prior.leaf_ids)
    for leaf in ("c", "f", "g"):
        assert np.all(w[leaves.index(leaf)] == 0.0)
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-12)
    assert set(prior.varying_leaves) == {"a", "b", "d", "e"}


def test_listed_held_children_keep_their_stated_weight_in_every_draw() -> None:
    local = {**LOCAL, "o3": 0.2, "o1": 0.4, "o2": 0.4}
    prior = pr.make_prior(NODES, CHILDREN, local, {None: 10.0, "o1": 9.0, "o2": 9.0, "o3": 9.0},
                          held={None: ["o1"]})  # fmt: skip
    top = next(g for g in prior.groups if g.parent_id is None)
    assert top.children == ("o2", "o3") and top.held == ("o1",)
    assert top.share == pytest.approx(0.6)
    local_draws = prior.local(prior.sample_ys(RNG, 300))
    assert np.all(local_draws[NODES.index("o1")] == 0.4)
    assert np.allclose(local_draws[NODES.index("o2")] + local_draws[NODES.index("o3")], 0.6)


def test_a_group_with_one_free_child_is_fixed() -> None:
    local = {**LOCAL, "e": 0.0, "d": 1.0}
    prior = pr.make_prior(NODES, CHILDREN, local, {None: 10.0, "o1": 9.0})
    assert ("o2", "fewer than two free children") in [(f.parent_id, f.reason) for f in prior.fixed]


def test_a_missing_concentration_is_refused() -> None:
    with pytest.raises(ValueError, match="no concentration"):
        pr.make_prior(NODES, CHILDREN, LOCAL, {None: 10.0})


def test_the_centre_is_the_stated_global_weights_and_the_prior_mean() -> None:
    prior = sy.bench_prior(47.0)
    expected = {"a1": 0.3, "a2": 0.18, "a3": 0.12, "b1": 0.28, "b2": 0.12}
    centre = prior.centre_global()
    mean, sd = prior.moments()
    for leaf, value in expected.items():
        assert centre[leaf] == pytest.approx(value, abs=1e-12)
        assert mean[leaf] == pytest.approx(value, abs=1e-12)
    draws = prior.global_weights(prior.sample_ys(RNG, 400_000))
    for i, leaf in enumerate(prior.leaf_ids):
        assert draws[i].mean() == pytest.approx(mean[leaf], abs=6 * sd[leaf] / math.sqrt(400_000))
        assert draws[i].std() == pytest.approx(sd[leaf], rel=0.01)


def test_prior_log_density_is_the_sum_of_independent_group_densities() -> None:
    prior = sy.bench_prior({None: 5.0, "obj_a": 7.0, "obj_b": 11.0})
    ys = [RNG.normal(size=(6, g.dim)) for g in prior.groups]
    total = sum(
        pr.group_log_density(y, g.centre, g.kappa) for g, y in zip(prior.groups, ys, strict=True)
    )
    assert np.allclose(prior.log_density(ys), total)
