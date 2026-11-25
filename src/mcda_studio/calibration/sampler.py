"""Adaptive random-walk Metropolis within Gibbs blocks, several chains at once (NumPy only).

Blocks: one per varying Dirichlet group (its ALR coordinates), in the prior's order, then
``(a, log beta)`` with ``a = alpha + beta * centre`` (the mean score at the prior centre), which
removes most of the intercept-slope correlation (Jacobian 1). Each iteration updates every block
d times, d being its dimension, so mixing per iteration does not fall with group size. During
warm-up each block's proposal covariance is re-estimated at iterations 200, 400, 800, ... from the
second half of the warm-up so far, and its scale follows a Robbins-Monro step toward a target
acceptance rate. After warm-up the kernel is fixed, so kept draws come from a Markov chain whose
stationary distribution is the posterior.

Chains are evaluated together as columns; chain ``c`` draws from ``PCG64(seed).jumped(c)``. Per
iteration and chain the stream gives, in order, the standard normals for every step of every block
and then one uniform per step. Design: ``docs/design/calibration.md`` section 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mcda_studio.calibration.likelihood import OutcomeData, log_likelihood, scores
from mcda_studio.methods.nested_dirichlet import NestedDirichletPrior

Array = NDArray[np.float64]

ALGORITHM = "calibration.rwm-gibbs.v1"
LOG_BETA_PRIOR_SD = 1.0  # beta ~ LogNormal(0, 1)
ALPHA_PRIOR_SD = 1.5
_ADAPT_FROM = 200
_RM_EXPONENT = 0.6
_JITTER = 1e-6


def alpha_prior_mean(base_rate: float) -> float:
    """``log(-log(1 - p0))``: the log hazard per window with event probability ``p0``."""
    if not 0.0 < base_rate < 1.0:
        raise ValueError("the base rate must lie strictly between 0 and 1")
    return math.log(-math.log1p(-base_rate))


@dataclass(frozen=True)
class SamplerSettings:
    chains: int = 4
    warmup: int = 2000
    draws: int = 3000  # kept iterations per chain
    samples: int = 4000  # stored vectors, thinned evenly from chains * draws
    seed: int = 0

    def check(self) -> None:
        if self.chains < 2:
            raise ValueError("at least two chains are needed to judge convergence")
        if self.warmup < 0 or self.draws < 4:
            raise ValueError("warm-up must be non-negative and at least four draws kept")
        if not 1 <= self.samples <= self.chains * self.draws:
            raise ValueError(
                f"cannot store {self.samples} samples from {self.chains * self.draws} kept "
                "draws; draws are never recycled"
            )
        if self.seed < 0:
            raise ValueError("the seed must be a non-negative integer")


@dataclass(frozen=True)
class Problem:
    prior: NestedDirichletPrior
    data: OutcomeData | None
    alpha_mean: float
    alpha_sd: float = ALPHA_PRIOR_SD
    log_beta_sd: float = LOG_BETA_PRIOR_SD

    def score_centre(self) -> float:
        """The mean score of the entering cases at the prior's centre weights (0 without data).

        The sampler moves ``a = alpha + beta * centre`` instead of ``alpha``. The map
        ``(a, log beta) -> (alpha, log beta)`` has Jacobian 1, so the posterior is unchanged; it
        only removes most of the correlation between the intercept and the slope.
        """
        if self.data is None or self.data.cases == 0:
            return 0.0
        centre = self.prior.centre_global()
        weights = np.array([[centre[leaf]] for leaf in self.prior.leaf_ids])
        return float(scores(self.data, weights).mean())


@dataclass
class Chains:
    """Kept draws: weights ``(chains, draws, leaves)``; alpha and beta ``(chains, draws)``."""

    weights: Array
    alpha: Array
    beta: Array
    acceptance: Array  # (blocks, chains), over kept iterations
    block_names: list[str] = field(default_factory=list)


def _normal_logpdf(x: Array, mean: float, sd: float) -> Array:
    out: Array = -0.5 * ((x - mean) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2.0 * math.pi)
    return out


def alpha_of(theta: Array, centre: float) -> Array:
    """``alpha = a - beta * centre`` from the sampled ``(a, log beta)``."""
    out: Array = theta[:, 0] - np.exp(theta[:, 1]) * centre
    return out


def log_posterior(
    problem: Problem, ys: list[Array], theta: Array, centre: float | None = None
) -> Array:
    """``(chains,)`` log posterior; ``ys[g]`` is ``(chains, dim_g)``; ``theta`` is ``(chains, 2)``
    holding ``(a, log beta)`` with ``a = alpha + beta * centre`` (``centre`` 0: plain alpha)."""
    log_beta = theta[:, 1]
    alpha = alpha_of(theta, 0.0 if centre is None else centre)
    chains = theta.shape[0]
    weights = problem.prior.global_weights(ys, chains)
    total = (
        problem.prior.log_density(ys, chains)
        + _normal_logpdf(alpha, problem.alpha_mean, problem.alpha_sd)
        + _normal_logpdf(log_beta, 0.0, problem.log_beta_sd)
        + log_likelihood(problem.data, weights, alpha, np.exp(log_beta))
    )
    out: Array = np.where(np.isfinite(total), total, -np.inf)
    return out


def _target(dim: int) -> float:
    return 0.44 if dim == 1 else 0.30


def run(problem: Problem, settings: SamplerSettings) -> Chains:
    """Run every chain; return the kept draws and per-block acceptance rates."""
    settings.check()
    prior = problem.prior
    chains = settings.chains
    dims = [*prior.dims, 2]
    n_blocks = len(dims)
    steps = list(dims)  # a d-dimensional block takes d Metropolis steps per iteration
    total_normals = sum(d * s for d, s in zip(dims, steps, strict=True))
    total_uniforms = sum(steps)
    streams = [np.random.Generator(np.random.PCG64(settings.seed).jumped(c)) for c in range(chains)]
    # Starting points: ALR coordinates around the centre, alpha and log beta from their priors.
    starts = [prior.initial_ys(rng) for rng in streams]
    ys = [np.stack([s[g] for s in starts]) for g in range(len(prior.groups))]
    centre = problem.score_centre()
    starts_ab = np.array(
        [
            [
                problem.alpha_mean + problem.alpha_sd * rng.standard_normal(),
                problem.log_beta_sd * rng.standard_normal(),
            ]
            for rng in streams
        ]
    )
    # Sample (a, log beta) with a = alpha + beta * centre.
    theta = np.stack([starts_ab[:, 0] + np.exp(starts_ab[:, 1]) * centre, starts_ab[:, 1]], axis=1)
    current = log_posterior(problem, ys, theta, centre)
    # Proposal state per block and chain.
    chol = [np.stack([np.eye(d) * math.sqrt(0.1)] * chains) for d in dims]
    log_scale = [np.full(chains, math.log(2.38 / math.sqrt(d))) for d in dims]
    history: list[list[Array]] = [[] for _ in dims]
    total_iterations = settings.warmup + settings.draws
    kept_w = np.empty((chains, settings.draws, len(prior.leaf_ids)))
    kept_a = np.empty((chains, settings.draws))
    kept_b = np.empty((chains, settings.draws))
    accepted = np.zeros((n_blocks, chains))
    checkpoint = _ADAPT_FROM

    def block_state(b: int) -> Array:
        return ys[b] if b < len(ys) else theta

    for it in range(total_iterations):
        warm = it < settings.warmup
        normals = np.stack([rng.standard_normal(total_normals) for rng in streams])
        uniforms = np.stack([rng.random(total_uniforms) for rng in streams])
        offset = 0
        u_index = 0
        for b, d in enumerate(dims):
            for _ in range(steps[b]):
                z = normals[:, offset : offset + d]
                offset += d
                step = np.einsum("cij,cj->ci", chol[b], z) * np.exp(log_scale[b])[:, None]
                proposal_ys = list(ys)
                proposal_theta = theta
                if b < len(ys):
                    proposal_ys[b] = ys[b] + step
                else:
                    proposal_theta = theta + step
                proposed = log_posterior(problem, proposal_ys, proposal_theta, centre)
                with np.errstate(invalid="ignore"):
                    accept = np.log(uniforms[:, u_index]) < proposed - current
                u_index += 1
                accept &= np.isfinite(proposed)
                if b < len(ys):
                    ys[b] = np.where(accept[:, None], proposal_ys[b], ys[b])
                else:
                    theta = np.where(accept[:, None], proposal_theta, theta)
                current = np.where(accept, proposed, current)
                if warm:
                    rate = 1.0 / (it + 1.0) ** _RM_EXPONENT
                    log_scale[b] = log_scale[b] + rate * (accept.astype(float) - _target(d))
                else:
                    accepted[b] += accept
            if warm:
                history[b].append(block_state(b).copy())
        if warm and it + 1 == checkpoint and checkpoint < settings.warmup:
            for b, d in enumerate(dims):
                recent = np.stack(history[b][checkpoint // 2 :], axis=1)  # (chains, n, d)
                for c in range(chains):
                    cov = np.atleast_2d(np.cov(recent[c], rowvar=False)) + _JITTER * np.eye(d)
                    try:
                        chol[b][c] = np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        chol[b][c] = np.eye(d) * math.sqrt(0.1)
                log_scale[b] = np.full(chains, math.log(2.38 / math.sqrt(d)))
            checkpoint *= 2
        if not warm:
            k = it - settings.warmup
            kept_w[:, k, :] = prior.global_weights(ys, chains).T
            kept_a[:, k] = alpha_of(theta, centre)
            kept_b[:, k] = np.exp(theta[:, 1])
    names = [
        "top level" if g.parent_id is None else f"group {g.parent_id}" for g in prior.groups
    ] + ["alpha, log beta"]
    return Chains(
        weights=kept_w,
        alpha=kept_a,
        beta=kept_b,
        acceptance=accepted / (max(1, settings.draws) * np.array(steps, dtype=float)[:, None]),
        block_names=names,
    )


def thin_indices(chains: int, draws: int, samples: int) -> list[tuple[int, int]]:
    """``samples`` evenly spaced (chain, iteration) pairs over the chains laid end to end."""
    total = chains * draws
    if not 1 <= samples <= total:
        raise ValueError("cannot store more samples than kept draws; draws are never recycled")
    picks = [0] if samples == 1 else [i * (total - 1) // (samples - 1) for i in range(samples)]
    return [(p // draws, p % draws) for p in picks]
