"""Revealed preference on synthetic decisions: the method recovers planted weights (DD8, Layer 2).

- Recovery: on the DD5 two-objective, five-leaf bench, the mean absolute error of the posterior-mean
  global weights falls with the number of decisions K in {50, 100, 200, 500, 2000}. The error at
  K = 200 is printed against the methodology notebook's "about 5%"; it is not a pass threshold.
- Coverage: with truths drawn from the prior, 90% intervals of every weight, alpha and beta cover
  at a rate whose 99% Clopper-Pearson interval contains 0.90.
- Simulation-based calibration (Talts et al. 2018): the rank of the truth among thinned posterior
  draws is uniform (chi-square, alpha = 0.01) for one weight, alpha and beta.
- Label noise: with a stated share of decisions flipped, beta attenuates (its 99% interval lies
  below the planted value and the clean fit's mean) while the weights are not biased (every planted
  weight stays inside the noisy fit's 99% interval). The noisy point estimates against the clean
  fit's 90% intervals are printed, not asserted.

Every run uses fixed seeds. These claims are about the method on synthetic data only; they say
nothing about any real decider.
"""

import numpy as np
import pytest
from scipy import stats

from mcda_studio.calibration import choices as ch
from mcda_studio.calibration import report
from mcda_studio.calibration import synthetic as sy
from mcda_studio.calibration.sampler import SamplerSettings, run, thin_indices
from mcda_studio.methods.smaa.intervals import clopper_pearson

pytestmark = [pytest.mark.regression, pytest.mark.slow]

FLIP_SHARE = 0.2


def stream(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def settings(seed: int) -> SamplerSettings:
    return SamplerSettings(warmup=1000, draws=2000, samples=4000, seed=seed)


def fit(prior, data, seed: int, sampler_settings: SamplerSettings | None = None):
    problem = ch.choice_problem(prior, data, sy.BASE_ACCEPT_RATE)
    return run(problem, sampler_settings or settings(seed))


def test_recovery_error_falls_with_the_number_of_decisions() -> None:
    prior = sy.bench_prior(10.0)
    truth = sy.planted_choice()
    target = np.array([truth.global_weights(prior)[leaf] for leaf in prior.leaf_ids])
    errors = {}
    for k in (50, 100, 200, 500, 2000):
        per_rep = []
        for rep in range(4):
            dset = sy.simulate_decisions(prior, truth, k, stream(1000 * k + rep))
            chains = fit(prior, dset.data, rep)
            assert report.diagnose(prior, chains).converged, (k, rep)
            per_rep.append(float(np.abs(chains.weights.mean(axis=(0, 1)) - target).mean()))
        errors[k] = float(np.mean(per_rep))
    print(f"\nrecovery: mean absolute error by K {errors}")
    print(f"K = 200: {errors[200]:.4f} (the notebook's figure is about 0.05)")
    assert errors[50] > errors[100] > errors[200] > errors[500] > errors[2000], errors
    assert errors[2000] < 0.5 * errors[50], errors


def test_ninety_percent_intervals_cover_truths_drawn_from_the_prior() -> None:
    prior = sy.bench_prior(20.0)
    hits = trials = 0
    for sim in range(40):
        rng = stream(500 + sim)
        truth = sy.draw_choice_truth(prior, rng)
        dset = sy.simulate_decisions(prior, truth, 200, rng)
        chains = fit(prior, dset.data, sim)
        assert report.diagnose(prior, chains).converged, sim
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
    print(f"\ncoverage: {hits}/{trials} = {hits / trials:.3f} (99% CI {low:.3f} to {high:.3f})")
    assert low <= 0.90 <= high, (hits, trials, low, high)


def test_simulation_based_calibration_ranks_are_uniform() -> None:
    prior = sy.bench_prior(20.0)
    sims, draws_kept = 100, 99
    ranks: dict[str, list[int]] = {"w[b1]": [], "alpha": [], "beta": []}
    column = list(prior.leaf_ids).index("b1")
    for sim in range(sims):
        rng = stream(9000 + sim)
        truth = sy.draw_choice_truth(prior, rng)
        dset = sy.simulate_decisions(prior, truth, 50, rng)
        chains = fit(prior, dset.data, sim)
        picks = thin_indices(4, 2000, draws_kept)
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
        print(f"\nSBC {name}: counts {counts.tolist()}, p = {p_value:.3f}")
        assert p_value > 0.01, (name, counts.tolist(), p_value)


@pytest.fixture(scope="module")
def noise_fits():
    prior = sy.bench_prior(10.0)
    truth = sy.planted_choice()
    clean = sy.simulate_decisions(prior, truth, 1000, stream(77))
    noisy = sy.simulate_decisions(prior, truth, 1000, stream(77), flip_share=FLIP_SHARE)
    return prior, truth, noisy, fit(prior, clean.data, 7), fit(prior, noisy.data, 7)


def test_label_noise_attenuates_beta(noise_fits) -> None:
    prior, truth, noisy, clean_chains, noisy_chains = noise_fits
    assert report.diagnose(prior, clean_chains).converged
    assert report.diagnose(prior, noisy_chains).converged
    assert noisy.flipped > 0
    low, high = np.quantile(noisy_chains.beta, [0.005, 0.995])
    clean_mean = float(clean_chains.beta.mean())
    print(
        f"\nlabel noise ({noisy.flipped} of 1000 flipped): beta clean mean {clean_mean:.2f}, "
        f"noisy mean {noisy_chains.beta.mean():.2f} (99% {low:.2f} to {high:.2f}), "
        f"planted {truth.beta}"
    )
    assert high < truth.beta
    assert high < clean_mean


def test_label_noise_leaves_the_weights_unbiased(noise_fits) -> None:
    prior, truth, _noisy, clean_chains, noisy_chains = noise_fits
    planted = truth.global_weights(prior)
    for i, leaf in enumerate(prior.leaf_ids):
        noisy_draws = noisy_chains.weights[:, :, i]
        low99, high99 = np.quantile(noisy_draws, [0.005, 0.995])
        clean_low, clean_high = np.quantile(clean_chains.weights[:, :, i], [0.05, 0.95])
        point = float(noisy_draws.mean())
        clean_mean = float(clean_chains.weights[:, :, i].mean())
        print(
            f"\n{leaf}: planted {planted[leaf]:.3f}, clean mean "
            f"{clean_mean:.3f} (90% {clean_low:.3f} to {clean_high:.3f}), "
            f"noisy mean {point:.3f} ({'inside' if clean_low <= point <= clean_high else 'outside'}"
            f" the clean 90% interval; noisy 99% {low99:.3f} to {high99:.3f})"
        )
        assert low99 <= planted[leaf] <= high99, leaf
