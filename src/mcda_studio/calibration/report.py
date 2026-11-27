"""Summaries of a sampler run: diagnostics per quantity, prior against posterior per leaf.

Shared by the service and the synthetic bench, so both apply the same convergence gate.
"""

from __future__ import annotations

import math

import numpy as np

from mcda_studio.calibration import diagnostics as dg
from mcda_studio.calibration.sampler import Chains
from mcda_studio.domain.common import DomainModel
from mcda_studio.methods.nested_dirichlet import NestedDirichletPrior

#: A run where every varying leaf keeps at least this share of its prior SD is prior-dominated.
PRIOR_DOMINATED_RATIO = 0.9
INTERVAL = 0.90


class QuantityDiagnostics(DomainModel):
    name: str
    rhat: float | None  # None when not finite (reported, and the run is not converged)
    ess_bulk: float | None


class Diagnostics(DomainModel):
    converged: bool
    rhat_max: float = dg.RHAT_MAX
    ess_min: float = dg.ESS_MIN
    quantities: list[QuantityDiagnostics]
    acceptance: dict[str, list[float]]  # block -> per chain


class LeafSummary(DomainModel):
    criterion_id: str
    varies: bool
    rated_cases: int | None = None
    prior_mean: float
    prior_sd: float
    posterior_mean: float
    posterior_sd: float
    interval: tuple[float, float]  # central 90%
    sd_ratio: float | None  # posterior SD / prior SD; None where the prior does not vary


class ParameterSummary(DomainModel):
    name: str
    prior_mean: float
    prior_sd: float
    posterior_mean: float
    posterior_sd: float
    interval: tuple[float, float]


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def diagnose(prior: NestedDirichletPrior, chains: Chains) -> Diagnostics:
    """R-hat and bulk ESS for every varying global weight, alpha and beta."""
    rows: list[QuantityDiagnostics] = []
    rhats: list[float] = []
    esses: list[float] = []
    columns = {leaf: i for i, leaf in enumerate(prior.leaf_ids)}
    series = [(f"w[{leaf}]", chains.weights[:, :, columns[leaf]]) for leaf in prior.varying_leaves]
    series += [("alpha", chains.alpha), ("beta", chains.beta)]
    for name, draws in series:
        r = dg.split_rhat(draws)
        e = dg.ess_bulk(draws)
        rhats.append(r)
        esses.append(e)
        rows.append(QuantityDiagnostics(name=name, rhat=_finite(r), ess_bulk=_finite(e)))
    return Diagnostics(
        converged=dg.converged(rhats, esses),
        quantities=rows,
        acceptance={
            name: [float(a) for a in chains.acceptance[b]]
            for b, name in enumerate(chains.block_names)
        },
    )


def _interval(draws: np.ndarray) -> tuple[float, float]:
    tail = (1.0 - INTERVAL) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(low), float(high)


def leaf_summaries(
    prior: NestedDirichletPrior, chains: Chains, rated: dict[str, int] | None = None
) -> list[LeafSummary]:
    mean, sd = prior.moments()
    varying = set(prior.varying_leaves)
    out = []
    for i, leaf in enumerate(prior.leaf_ids):
        draws = chains.weights[:, :, i].ravel()
        post_sd = float(draws.std(ddof=1))
        out.append(
            LeafSummary(
                criterion_id=leaf,
                varies=leaf in varying,
                rated_cases=None if rated is None else rated.get(leaf, 0),
                prior_mean=mean[leaf],
                prior_sd=sd[leaf],
                posterior_mean=float(draws.mean()),
                posterior_sd=post_sd,
                interval=_interval(draws),
                sd_ratio=post_sd / sd[leaf] if leaf in varying and sd[leaf] > 0 else None,
            )
        )
    return out


def parameter_summaries(
    chains: Chains,
    alpha_mean: float,
    alpha_sd: float,
    log_beta_sd: float,
    log_beta_mean: float = 0.0,
) -> list[ParameterSummary]:
    beta_mean = math.exp(log_beta_mean + log_beta_sd**2 / 2.0)
    beta_sd = math.sqrt(
        (math.exp(log_beta_sd**2) - 1.0) * math.exp(2.0 * log_beta_mean + log_beta_sd**2)
    )
    out = []
    for name, draws, pm, ps in (
        ("alpha", chains.alpha.ravel(), alpha_mean, alpha_sd),
        ("beta", chains.beta.ravel(), beta_mean, beta_sd),
    ):
        out.append(
            ParameterSummary(
                name=name,
                prior_mean=pm,
                prior_sd=ps,
                posterior_mean=float(draws.mean()),
                posterior_sd=float(draws.std(ddof=1)),
                interval=_interval(draws),
            )
        )
    return out


def prior_dominated(leaves: list[LeafSummary]) -> bool:
    ratios = [leaf.sd_ratio for leaf in leaves if leaf.sd_ratio is not None]
    return bool(ratios) and all(r > PRIOR_DOMINATED_RATIO for r in ratios)
