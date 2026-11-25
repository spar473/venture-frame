"""Convergence diagnostics: rank-normalised split-R-hat and bulk effective sample size.

Following Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021), "Rank-normalization, folding,
and localization: an improved R-hat for assessing convergence of MCMC", Bayesian Analysis 16(2).
Draws are arrays of shape ``(chains, iterations)``. Normal scores use the standard library's
``statistics.NormalDist`` (no SciPy at runtime).

A run is converged only when every reported R-hat is below ``RHAT_MAX`` and every bulk ESS is at
least ``ESS_MIN``. These are constants, not options.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

RHAT_MAX = 1.01
ESS_MIN = 400.0

_NORMAL = NormalDist()


def split_chains(draws: Array) -> Array:
    """Each chain cut into its first and last halves (a middle draw is dropped when odd)."""
    x = np.asarray(draws, dtype=np.float64)
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, x.shape[1] - half :]], axis=0)


def _average_ranks(values: Array) -> Array:
    """Ranks 1..n with ties given their average rank."""
    flat = values.ravel()
    order = np.argsort(flat, kind="mergesort")
    ordered = flat[order]
    first = np.r_[True, ordered[1:] != ordered[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], flat.size] - 1
    ranks = np.empty(flat.size)
    ranks[order] = ((starts + ends) / 2.0 + 1.0)[np.cumsum(first) - 1]
    return ranks.reshape(values.shape)


def rank_normalise(draws: Array) -> Array:
    """Normal scores of the pooled ranks: ``Phi^-1((r - 3/8) / (S + 1/4))``."""
    x = np.asarray(draws, dtype=np.float64)
    ranks = _average_ranks(x)
    size = x.size
    z = np.array([_NORMAL.inv_cdf((r - 0.375) / (size + 0.25)) for r in ranks.ravel()])
    return z.reshape(x.shape)


def _rhat(chains: Array) -> float:
    """The classic potential scale reduction on ``(chains, n)`` draws."""
    n = chains.shape[1]
    means = chains.mean(axis=1)
    within = float(chains.var(axis=1, ddof=1).mean())
    between = n * float(means.var(ddof=1))
    if within == 0.0:
        return math.inf if between > 0.0 else math.nan
    pooled = (n - 1) / n * within + between / n
    return math.sqrt(pooled / within)


def split_rhat(draws: Array) -> float:
    """Rank-normalised split-R-hat: the larger of the bulk and the folded (tail) versions."""
    halves = split_chains(draws)
    bulk = _rhat(rank_normalise(halves))
    folded = _rhat(rank_normalise(np.abs(halves - np.median(halves))))
    if math.isnan(bulk) or math.isnan(folded):
        return math.nan
    return max(bulk, folded)


def _autocovariance(x: Array) -> Array:
    """Biased autocovariance of each row by FFT, ``(chains, n)``."""
    n = x.shape[1]
    centred = x - x.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)[:, :n]
    out: Array = acov / n
    return out


def ess(chains: Array) -> float:
    """Effective sample size of ``(chains, n)`` draws (Geyer's initial monotone sequence)."""
    m, n = chains.shape
    if n < 4:
        return math.nan
    acov = _autocovariance(chains)
    chain_mean = chains.mean(axis=1)
    mean_var = float(acov[:, 0].mean()) * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n
    if m > 1:
        var_plus += float(chain_mean.var(ddof=1))
    if var_plus == 0.0:
        return math.nan
    rho = np.zeros(n)
    rho[0] = 1.0
    even = 1.0
    odd = 1.0 - (mean_var - float(acov[:, 1].mean())) / var_plus
    rho[1] = odd
    t = 1
    while t < n - 3 and even + odd > 0.0:
        even = 1.0 - (mean_var - float(acov[:, t + 1].mean())) / var_plus
        odd = 1.0 - (mean_var - float(acov[:, t + 2].mean())) / var_plus
        if even + odd >= 0.0:
            rho[t + 1] = even
            rho[t + 2] = odd
        t += 2
    max_t = t - 2
    if even > 0.0:
        rho[max_t + 1] = even
    t = 1
    while t <= max_t - 2:  # make the paired sums monotone
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2
    total = float(m * n)
    tau = -1.0 + 2.0 * float(rho[: max_t + 1].sum()) + float(rho[max_t + 1 : max_t + 2].sum())
    tau = max(tau, 1.0 / math.log10(total))
    return total / tau


def ess_bulk(draws: Array) -> float:
    """Bulk ESS: the ESS of the rank-normalised split chains."""
    return ess(rank_normalise(split_chains(draws)))


def converged(rhats: list[float], esses: list[float]) -> bool:
    """Every R-hat finite and below ``RHAT_MAX``, every ESS finite and at least ``ESS_MIN``."""
    return all(math.isfinite(r) and r < RHAT_MAX for r in rhats) and all(
        math.isfinite(e) and e >= ESS_MIN for e in esses
    )
