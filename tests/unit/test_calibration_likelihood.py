"""Calibration outcome model: each status against a hand-computed log-likelihood, the interval form
for partial dates, competing events as censoring, and scores that renormalise over rated leaves.
"""

import math

import numpy as np
import pytest

from mcda_studio.calibration.likelihood import (
    OutcomeData,
    case_log_likelihood,
    encode,
    log1mexp,
    log_likelihood,
    scores,
)

ALPHA, BETA = -1.3, 2.2


def one_case(observation, values=(0.5, 1.0), rated=(1.0, 1.0)) -> OutcomeData:
    return OutcomeData.build(np.array([values]), np.array([rated]), [observation], [0, 1])


def ll(data: OutcomeData, weights=(0.25, 0.75)) -> float:
    w = np.array(weights)[:, None]
    return float(case_log_likelihood(data, w, np.array([ALPHA]), np.array([BETA]))[0, 0])


S = (0.25 * 0.5 + 0.75 * 1.0) / 1.0  # the score of the default case
LAM = math.exp(ALPHA + BETA * S)


def test_event_on_an_exact_day_is_the_exponential_density() -> None:
    assert ll(one_case(encode("event", time=0.4))) == pytest.approx(
        math.log(LAM) - LAM * 0.4, abs=1e-12
    )
    assert ll(one_case(encode("event", time=0.0))) == pytest.approx(math.log(LAM), abs=1e-12)


def test_no_event_is_survival_through_the_window() -> None:
    assert ll(one_case(encode("no_event"))) == pytest.approx(-LAM, abs=1e-12)


def test_censored_is_survival_to_the_censoring_time() -> None:
    assert ll(one_case(encode("censored", time=0.3))) == pytest.approx(-LAM * 0.3, abs=1e-12)


def test_competing_event_equals_censored_at_the_same_time() -> None:
    for t in (0.0, 0.17, 0.5, 0.99):
        assert ll(one_case(encode("competing_event", time=t))) == ll(
            one_case(encode("censored", time=t))
        )
    # On a partial date it is censored at the earliest day the date can mean.
    assert ll(one_case(encode("competing_event", interval=(0.25, 0.5)))) == ll(
        one_case(encode("censored", time=0.25))
    )


def test_partial_event_date_is_the_probability_of_the_interval() -> None:
    a, b = 0.25, 1 / 3
    expected = math.log(math.exp(-LAM * a) - math.exp(-LAM * b))
    assert ll(one_case(encode("event", interval=(a, b)))) == pytest.approx(expected, abs=1e-12)


def test_interval_is_clipped_to_the_window() -> None:
    clipped = ll(one_case(encode("event", interval=(-0.2, 0.1))))
    assert clipped == pytest.approx(math.log(1 - math.exp(-LAM * 0.1)), abs=1e-12)
    with pytest.raises(ValueError, match="positive interval"):
        encode("event", interval=(1.2, 1.5))


def test_a_narrow_interval_approaches_density_times_width() -> None:
    t, h = 0.4, 1e-7
    narrow = ll(one_case(encode("event", interval=(t, t + h))))
    assert narrow == pytest.approx(math.log(LAM) - LAM * t + math.log(h), abs=1e-6)


def test_log1mexp_is_accurate_at_both_ends() -> None:
    for x in (1e-12, 1e-6, 0.1, math.log(2.0), 1.0, 30.0, 700.0):
        exact = math.log(-math.expm1(-x)) if x < 1 else math.log1p(-math.exp(-x))
        assert float(log1mexp(np.array([x]))[0]) == pytest.approx(exact, rel=1e-14, abs=1e-300)


def test_censored_is_not_no_event_and_competing_is_not_event() -> None:
    assert ll(one_case(encode("censored", time=0.3))) != ll(one_case(encode("no_event")))
    assert ll(one_case(encode("competing_event", time=0.3))) != ll(
        one_case(encode("event", time=0.3))
    )


def test_scores_renormalise_over_rated_leaves_and_missing_is_not_zero() -> None:
    values = np.array([[0.5, 1.0], [0.5, 0.0], [0.5, 0.0]])
    rated = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    data = OutcomeData.build(values, rated, [encode("no_event")] * 3, [0, 1])
    s = scores(data, np.array([[0.25], [0.75]]))[:, 0]
    assert s[0] == pytest.approx(0.875)
    assert s[1] == pytest.approx(0.5)  # only the rated leaf counts
    assert s[2] == pytest.approx(0.125)  # a rated 0 is a value, not a gap
    assert s[1] != s[2]


def test_the_total_is_the_sum_of_cases_and_chains_are_independent_columns() -> None:
    obs = [encode("event", time=0.2), encode("no_event"), encode("censored", time=0.6)]
    data = OutcomeData.build(np.array([[0.0, 1.0], [1.0, 0.5], [0.25, 0.25]]),
                             np.ones((3, 2)), obs, [0, 1])  # fmt: skip
    w = np.array([[0.3, 0.6], [0.7, 0.4]])
    alpha, beta = np.array([-1.0, 0.5]), np.array([1.0, 3.0])
    total = log_likelihood(data, w, alpha, beta)
    for c in range(2):
        alone = log_likelihood(data, w[:, c : c + 1], alpha[c : c + 1], beta[c : c + 1])
        assert total[c] == pytest.approx(alone[0], abs=1e-12)
    assert np.all(log_likelihood(None, w, alpha, beta) == 0.0)


def test_every_entering_case_rates_a_leaf() -> None:
    with pytest.raises(ValueError, match="rates at least one"):
        OutcomeData.build(np.zeros((1, 2)), np.zeros((1, 2)), [encode("no_event")], [0, 1])


def test_unknown_does_not_enter_the_likelihood() -> None:
    with pytest.raises(ValueError, match="does not enter"):
        encode("unknown")  # type: ignore[arg-type]
