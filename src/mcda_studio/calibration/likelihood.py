"""The outcome model: a constant hazard within the window, censoring-aware.

``lambda_i = exp(alpha + beta * s_i)`` in events per window, with time in window fractions and the
score ``s_i`` the case's rated values weighted by the global weights renormalised over the leaves
it rates (missing is never zero). Each entering case contributes:

==================  ================================================================
event, exact day t   ``log lambda - lambda t``
event, partial date  ``-lambda a + log(1 - exp(-lambda (b - a)))``
no_event             ``-lambda``
censored at c        ``-lambda c``
competing event, t   ``-lambda t`` (censored at t: a cause-specific hazard)
==================  ================================================================

So every case is ``density * eta - lambda * exposure + [width > 0] log(1 - exp(-lambda width))``
with ``eta = log lambda``. Design: ``docs/design/calibration.md`` section 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Status = Literal["event", "no_event", "censored", "competing_event"]


@dataclass(frozen=True)
class Observation:
    """One entering case's outcome, encoded for the likelihood."""

    density: float  # 1.0 for an event on an exact day, else 0.0
    exposure: float  # the time, in window fractions, the case is known to be event-free
    width: float  # the length of a partial event date's interval, else 0.0


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def encode(
    status: Status,
    *,
    time: float | None = None,
    interval: tuple[float, float] | None = None,
) -> Observation:
    """Encode one outcome. ``time`` is the event, competing-event or censoring time; an event or
    competing event on a partial date gives ``interval`` (``a``, ``b``) instead. Times are window
    fractions and are clipped to ``[0, 1]``."""
    if status == "no_event":
        return Observation(0.0, 1.0, 0.0)
    if status == "censored":
        if time is None:
            raise ValueError("censored needs the time it was watched to")
        return Observation(0.0, _clip(time), 0.0)
    if status == "competing_event":
        if interval is not None:
            return Observation(0.0, _clip(interval[0]), 0.0)  # censored at the earliest day
        if time is None:
            raise ValueError("a competing event needs its time")
        return Observation(0.0, _clip(time), 0.0)
    if status == "event":
        if interval is not None:
            a, b = _clip(interval[0]), _clip(interval[1])
            if not b > a:
                raise ValueError("a partial event date needs a positive interval")
            return Observation(0.0, a, b - a)
        if time is None:
            raise ValueError("an event needs its time")
        return Observation(1.0, _clip(time), 0.0)
    raise ValueError(f"status {status!r} does not enter the likelihood")


@dataclass(frozen=True)
class OutcomeData:
    """Entering cases: values on the compared leaves and encoded outcomes.

    ``values`` and ``rated`` are ``(cases, compared leaves)``; a leaf a case does not rate has
    ``rated = 0`` and its value is ignored. ``leaf_index`` maps each compared leaf to its position
    in the prior's ``leaf_ids``.
    """

    values: Array
    rated: Array
    density: Array
    exposure: Array
    width: Array
    leaf_index: NDArray[np.intp]

    @property
    def cases(self) -> int:
        return int(self.values.shape[0])

    @staticmethod
    def build(
        values: Array, rated: Array, observations: list[Observation], leaf_index: list[int]
    ) -> OutcomeData:
        rated = np.asarray(rated, dtype=np.float64)
        if np.any(rated.sum(axis=1) == 0) and rated.shape[0]:
            raise ValueError("every entering case rates at least one compared leaf")
        return OutcomeData(
            values=np.where(rated > 0, np.asarray(values, dtype=np.float64), 0.0),
            rated=rated,
            density=np.array([o.density for o in observations], dtype=np.float64),
            exposure=np.array([o.exposure for o in observations], dtype=np.float64),
            width=np.array([o.width for o in observations], dtype=np.float64),
            leaf_index=np.asarray(leaf_index, dtype=np.intp),
        )


def scores(data: OutcomeData, global_weights: Array) -> Array:
    """``s_i`` per case and chain, ``(cases, chains)``: weights renormalised over rated leaves."""
    w = global_weights[data.leaf_index, :]  # (compared, chains)
    num = np.zeros((data.cases, w.shape[1]))
    den = np.zeros((data.cases, w.shape[1]))
    for j in range(w.shape[0]):  # left to right, one leaf at a time
        num = num + (data.values[:, j : j + 1] * data.rated[:, j : j + 1]) * w[j]
        den = den + data.rated[:, j : j + 1] * w[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num / den
    # A case whose rated leaves all have weight 0 has no score: it carries no information.
    out: Array = np.where(den > 0.0, s, 0.0)
    return out


def log1mexp(x: Array) -> Array:
    """``log(1 - exp(-x))`` for ``x > 0`` (Maechler 2012): accurate at both ends."""
    x = np.asarray(x, dtype=np.float64)
    small = x <= math.log(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: Array = np.where(
            small, np.log(-np.expm1(-np.where(small, x, 1.0))), np.log1p(-np.exp(-x))
        )
    return out


def case_log_likelihood(
    data: OutcomeData, global_weights: Array, alpha: Array, beta: Array
) -> Array:
    """Each case's contribution, ``(cases, chains)``."""
    eta = alpha[None, :] + beta[None, :] * scores(data, global_weights)
    lam = np.exp(eta)
    out = data.density[:, None] * eta - lam * data.exposure[:, None]
    partial = data.width > 0.0
    if np.any(partial):
        out[partial] = out[partial] + log1mexp(lam[partial] * data.width[partial, None])
    result: Array = out
    return result


def log_likelihood(
    data: OutcomeData | None, global_weights: Array, alpha: Array, beta: Array
) -> Array:
    """The total log-likelihood per chain, ``(chains,)``; zero without data."""
    if data is None or data.cases == 0:
        return np.zeros(alpha.shape[0])
    total: Array = case_log_likelihood(data, global_weights, alpha, beta).sum(axis=0)
    return total
