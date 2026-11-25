"""Calibration sampler: with no data it returns the prior (KS against the exact Beta marginals, means
within six standard errors), it repeats for a seed, it never recycles draws, and a short run fails
the convergence gate.
"""

import math

import numpy as np
import pytest
from scipy import stats

from mcda_studio.calibration import diagnostics as dg
from mcda_studio.calibration import report
from mcda_studio.calibration import synthetic as sy
from mcda_studio.calibration.sampler import (
    Problem,
    SamplerSettings,
    alpha_prior_mean,
    run,
    thin_indices,
)

KAPPA = {None: 8.0, "obj_a": 6.0, "obj_b": 12.0}


@pytest.fixture(scope="module")
def prior_only():
    prior = sy.bench_prior(KAPPA)
    problem = Problem(prior, None, alpha_prior_mean(0.15))
    settings = SamplerSettings(chains=4, warmup=1000, draws=6000, samples=4000, seed=11)
    return prior, problem, run(problem, settings)


def _locals(prior, chains) -> dict[str, np.ndarray]:
    w = {leaf: chains.weights[:, :, i] for i, leaf in enumerate(prior.leaf_ids)}
    obj_a = w["a1"] + w["a2"] + w["a3"]
    obj_b = w["b1"] + w["b2"]
    return {
        "obj_a": obj_a, "obj_b": obj_b,
        "a1": w["a1"] / obj_a, "a2": w["a2"] / obj_a, "a3": w["a3"] / obj_a,
        "b1": w["b1"] / obj_b, "b2": w["b2"] / obj_b,
    }  # fmt: skip


def test_with_no_data_every_share_has_its_exact_beta_marginal(prior_only) -> None:
    prior, _problem, chains = prior_only
    shares = _locals(prior, chains)
    for group in prior.groups:
        for child, centre in zip(group.children, group.centre, strict=True):
            a, b = group.kappa * centre, group.kappa * (1 - centre)
            draws = shares[child]
            # Thin to near-independent draws for the KS test: every 30th iteration of each chain.
            thinned = draws[:, ::30].ravel()
            p_value = stats.kstest(thinned, stats.beta(a, b).cdf).pvalue
            assert p_value > 0.01, (child, p_value)
            mean_se = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))) / math.sqrt(
                dg.ess_bulk(draws)
            )
            assert abs(draws.mean() - a / (a + b)) < 6 * mean_se, child


def test_with_no_data_alpha_and_beta_keep_their_priors(prior_only) -> None:
    _prior, problem, chains = prior_only
    alpha = chains.alpha[:, ::30].ravel()
    assert stats.kstest(alpha, stats.norm(problem.alpha_mean, 1.5).cdf).pvalue > 0.01
    beta = chains.beta[:, ::30].ravel()
    assert stats.kstest(beta, stats.lognorm(1.0).cdf).pvalue > 0.01


def test_a_prior_only_run_converges_and_is_prior_dominated(prior_only) -> None:
    prior, _problem, chains = prior_only
    diagnostics = report.diagnose(prior, chains)
    assert diagnostics.converged
    leaves = report.leaf_summaries(prior, chains)
    assert report.prior_dominated(leaves)
    for leaf in leaves:
        assert leaf.sd_ratio == pytest.approx(1.0, abs=0.1)


def small(seed: int = 3, cases: int = 120) -> tuple:
    prior = sy.bench_prior(20.0)
    sset = sy.simulate(prior, sy.planted(), cases, np.random.Generator(np.random.PCG64(seed)))
    return prior, Problem(prior, sset.data, alpha_prior_mean(0.15))


def test_the_same_seed_gives_the_same_draws_and_another_seed_does_not() -> None:
    _prior, problem = small()
    settings = SamplerSettings(warmup=300, draws=200, samples=400, seed=5)
    first, second = run(problem, settings), run(problem, settings)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.alpha, second.alpha)
    other = run(problem, SamplerSettings(warmup=300, draws=200, samples=400, seed=6))
    assert not np.array_equal(first.weights, other.weights)


def test_chains_use_different_streams() -> None:
    _prior, problem = small()
    chains = run(problem, SamplerSettings(warmup=50, draws=50, samples=100, seed=0))
    assert not np.array_equal(chains.weights[0], chains.weights[1])


def test_a_short_run_fails_the_convergence_gate() -> None:
    prior, problem = small()
    chains = run(problem, SamplerSettings(warmup=25, draws=25, samples=100, seed=0))
    diagnostics = report.diagnose(prior, chains)
    assert not diagnostics.converged
    assert any(q.ess_bulk is None or q.ess_bulk < dg.ESS_MIN for q in diagnostics.quantities)


def test_draws_are_thinned_evenly_and_never_recycled() -> None:
    picks = thin_indices(4, 3000, 4000)
    assert len(picks) == len(set(picks)) == 4000
    assert picks[0] == (0, 0) and picks[-1] == (3, 2999)
    assert {c for c, _ in picks} == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="never recycled"):
        thin_indices(4, 100, 401)
    with pytest.raises(ValueError, match="never recycled"):
        SamplerSettings(chains=4, draws=100, samples=401).check()


def test_settings_are_checked() -> None:
    with pytest.raises(ValueError, match="two chains"):
        SamplerSettings(chains=1).check()
    with pytest.raises(ValueError, match="seed"):
        SamplerSettings(seed=-1).check()
    with pytest.raises(ValueError, match="strictly between"):
        alpha_prior_mean(1.0)
    assert alpha_prior_mean(0.15) == pytest.approx(math.log(-math.log(0.85)))


def test_beta_is_always_positive_and_weights_sum_to_one() -> None:
    _prior, problem = small()
    chains = run(problem, SamplerSettings(warmup=200, draws=200, samples=400, seed=1))
    assert np.all(chains.beta > 0)
    assert np.allclose(chains.weights.sum(axis=2), 1.0, atol=1e-12)
