"""Calibration on synthetic data: the method recovers planted weights (deep dive 5).

- Recovery: error against planted weights falls with the number of cases.
- Coverage: with truths drawn from the prior, 90% intervals cover at a rate whose 99%
  Clopper-Pearson interval contains 0.90.
- Simulation-based calibration (Talts et al. 2018): the rank of the truth among posterior draws is
  uniform (chi-square, alpha = 0.01) for one weight, alpha and beta.
- Censoring matters: with score-dependent follow-up, dropping censored cases and calling them no
  event both bias beta beyond their own 99% intervals; the censoring-aware likelihood covers it.

Every run uses fixed seeds. These claims are about the method on synthetic data only.
"""

import numpy as np
import pytest
from scipy import stats

from mcda_studio.calibration import report
from mcda_studio.calibration import synthetic as sy
from mcda_studio.calibration.sampler import (
    Problem,
    SamplerSettings,
    alpha_prior_mean,
    run,
    thin_indices,
)
from mcda_studio.methods.smaa.intervals import clopper_pearson

pytestmark = [pytest.mark.regression, pytest.mark.slow]

ALPHA_MEAN = alpha_prior_mean(sy.BASE_RATE)


def stream(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def fit(prior, data, seed: int, settings: SamplerSettings | None = None):
    return run(Problem(prior, data, ALPHA_MEAN), settings or SamplerSettings(seed=seed))


def test_recovery_error_falls_with_the_number_of_cases() -> None:
    prior = sy.bench_prior(10.0)
    truth = sy.planted()
    target = np.array([truth.global_weights(prior)[leaf] for leaf in prior.leaf_ids])
    errors = {}
    for n in (50, 150, 500, 2000):
        per_rep = []
        for rep in range(4):
            sset = sy.simulate(prior, truth, n, stream(1000 * n + rep))
            chains = fit(prior, sset.data, rep)
            assert report.diagnose(prior, chains).converged, (n, rep)
            per_rep.append(float(np.abs(chains.weights.mean(axis=(0, 1)) - target).mean()))
        errors[n] = float(np.mean(per_rep))
    assert errors[50] > errors[150] > errors[500] > errors[2000], errors
    assert errors[2000] < 0.5 * errors[50], errors


def test_ninety_percent_intervals_cover_truths_drawn_from_the_prior() -> None:
    prior = sy.bench_prior(20.0)
    settings_for = lambda seed: SamplerSettings(warmup=1000, draws=2000, samples=4000, seed=seed)  # noqa: E731
    hits = trials = 0
    for sim in range(40):
        rng = stream(500 + sim)
        truth = sy.draw_truth(prior, rng, ALPHA_MEAN)
        sset = sy.simulate(prior, truth, 150, rng)
        chains = fit(prior, sset.data, sim, settings_for(sim))
        planted = truth.global_weights(prior)
        for i, leaf in enumerate(prior.leaf_ids):
            low, high = np.quantile(chains.weights[:, :, i], [0.05, 0.95])
            hits += int(low <= planted[leaf] <= high)
            trials += 1
        for draws, value in ((chains.alpha, truth.alpha), (chains.beta, truth.beta)):
            low, high = np.quantile(draws, [0.05, 0.95])
            hits += int(low <= value <= high)
            trials += 1
    low, high = clopper_pearson(hits, trials, 0.99)
    assert low <= 0.90 <= high, (hits, trials, low, high)


def test_simulation_based_calibration_ranks_are_uniform() -> None:
    prior = sy.bench_prior(20.0)
    sims, draws_kept = 120, 99
    ranks: dict[str, list[int]] = {"w[b1]": [], "alpha": [], "beta": []}
    column = list(prior.leaf_ids).index("b1")
    for sim in range(sims):
        rng = stream(9000 + sim)
        truth = sy.draw_truth(prior, rng, ALPHA_MEAN)
        sset = sy.simulate(prior, truth, 50, rng)
        settings = SamplerSettings(warmup=1000, draws=2000, samples=4000, seed=sim)
        chains = fit(prior, sset.data, sim, settings)
        picks = thin_indices(settings.chains, settings.draws, draws_kept)
        values = {
            "w[b1]": (chains.weights[:, :, column], truth.global_weights(prior)["b1"]),
            "alpha": (chains.alpha, truth.alpha),
            "beta": (chains.beta, truth.beta),
        }
        for name, (draws, value) in values.items():
            thinned = np.array([draws[c, k] for c, k in picks])
            ranks[name].append(int((thinned < value).sum()))
    for name, values in ranks.items():
        counts = np.bincount(np.array(values) // 10, minlength=10)  # ranks 0..99 in ten bins
        assert counts.sum() == sims
        p_value = stats.chisquare(counts).pvalue
        assert p_value > 0.01, (name, counts.tolist(), p_value)


@pytest.fixture(scope="module")
def censored_fits():
    prior = sy.bench_prior(47.0)
    sset = sy.simulate(prior, sy.planted(), 2000, stream(1), censoring="score")
    fits = {
        "censoring-aware": fit(prior, sset.data, 1),
        "censored dropped": fit(prior, sy.censoring_variant(sset, "drop"), 1),
        "censored as no event": fit(prior, sy.censoring_variant(sset, "no_event"), 1),
    }
    return sset, fits


def test_the_bench_censors_heavily_and_by_score(censored_fits) -> None:
    sset, _fits = censored_fits
    share = sset.statuses.count("censored") / len(sset.statuses)
    assert share > 0.4


def test_censoring_aware_likelihood_recovers_beta_where_naive_treatments_do_not(
    censored_fits,
) -> None:
    _sset, fits = censored_fits
    beta = sy.PLANTED_BETA

    def interval(name: str) -> tuple[float, float]:
        low, high = np.quantile(fits[name].beta, [0.005, 0.995])
        return float(low), float(high)

    low, high = interval("censoring-aware")
    assert low <= beta <= high
    low, _ = interval("censored dropped")
    assert low > beta  # leaving out censored cases overstates beta
    _, high = interval("censored as no event")
    assert high < beta  # calling them no event understates it
