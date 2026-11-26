"""Calibration diagnostics: split-R-hat and bulk ESS on chains whose answers are known, and the
convergence gate at its exact thresholds.

Oracles: independent draws (R-hat near 1, ESS near the draw count), an AR(1) process whose
integrated autocorrelation time is (1 + rho) / (1 - rho), and chains stuck in different places.
"""

import math

import numpy as np
import pytest

from mcda_studio.calibration import diagnostics as dg

RNG = np.random.Generator(np.random.PCG64(7))


def ar1(rho: float, chains: int, n: int) -> np.ndarray:
    x = np.empty((chains, n))
    x[:, 0] = RNG.standard_normal(chains) / math.sqrt(1 - rho**2)
    noise = RNG.standard_normal((chains, n))
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + noise[:, t]
    return x


def test_independent_chains_have_rhat_near_one_and_ess_near_the_draw_count() -> None:
    draws = RNG.standard_normal((4, 2000))
    assert dg.split_rhat(draws) < 1.005
    assert dg.ess_bulk(draws) == pytest.approx(8000, rel=0.1)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_ess_of_an_ar1_process_matches_its_autocorrelation_time(rho: float) -> None:
    draws = ar1(rho, 4, 20_000)
    expected = 80_000 * (1 - rho) / (1 + rho)
    assert dg.ess_bulk(draws) == pytest.approx(expected, rel=0.15)


def test_chains_in_different_places_have_a_large_rhat() -> None:
    draws = RNG.standard_normal((4, 1000)) + np.array([[0.0], [0.0], [0.0], [2.0]])
    assert dg.split_rhat(draws) > 1.1


def test_a_trend_within_chains_is_caught_by_splitting() -> None:
    trend = np.linspace(0.0, 3.0, 1000)
    draws = RNG.standard_normal((4, 1000)) * 0.3 + trend
    assert dg.split_rhat(draws) > 1.1


def test_a_scale_difference_is_caught_by_the_folded_rhat() -> None:
    draws = RNG.standard_normal((4, 2000)) * np.array([[1.0], [1.0], [1.0], [3.0]])
    halves = dg.split_chains(draws)
    bulk = dg._rhat(dg.rank_normalise(halves))
    assert bulk < 1.01  # the means agree
    assert dg.split_rhat(draws) > 1.01  # the spreads do not


def test_stuck_chains_are_never_called_converged() -> None:
    draws = np.repeat(np.array([[0.1], [0.2], [0.3], [0.4]]), 100, axis=1)
    assert dg.split_rhat(draws) == math.inf
    assert not dg.converged([dg.split_rhat(draws)], [1000.0])
    constant = np.zeros((4, 100))
    assert math.isnan(dg.split_rhat(constant))
    assert not dg.converged([dg.split_rhat(constant)], [1000.0])


def test_ranks_average_ties_and_normal_scores_are_symmetric() -> None:
    ranks = dg._average_ranks(np.array([[3.0, 1.0, 3.0, 2.0]]))
    assert ranks.tolist() == [[3.5, 1.0, 3.5, 2.0]]
    z = dg.rank_normalise(np.arange(10.0).reshape(2, 5))
    assert np.allclose(z.ravel(), -z.ravel()[::-1])


def test_the_gate_uses_strict_and_inclusive_thresholds_exactly() -> None:
    assert dg.RHAT_MAX == 1.01 and dg.ESS_MIN == 400
    assert dg.converged([1.0099999], [400.0])
    assert not dg.converged([1.01], [10_000.0])
    assert not dg.converged([1.0], [399.999])
    assert not dg.converged([math.nan], [1000.0])
    assert not dg.converged([1.0], [math.nan])


def test_split_chains_cut_each_chain_into_halves() -> None:
    draws = np.arange(14.0).reshape(2, 7)
    halves = dg.split_chains(draws)
    assert halves.tolist() == [[0, 1, 2], [7, 8, 9], [4, 5, 6], [11, 12, 13]]
