"""Trade-off elicitation's noisy mode: a Bradley-Terry likelihood on the calibration sampler.

``P(A over B) = sigmoid(beta * w . (v(A) - v(B)))`` with ``w`` the global leaf weights. It plugs
into ``sampler.Problem.log_likelihood`` (the hook revealed preference added) and reuses the sampler
and its convergence gate (R-hat below 1.01, bulk ESS at least 400) unchanged. ``alpha`` plays no
part in the likelihood; it keeps a standard normal prior and is reported as a nuisance.

Priors: the stated weight prior (``methods/tradeoff.py::ordered_nested_prior``) and
``beta ~ LogNormal(log 20, 1)``. Offline and opt-in, reached only from the command line.
Design: ``docs/design/tradeoff-elicitation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mcda_studio.calibration import report, sampler
from mcda_studio.methods.nested_dirichlet import NestedDirichletPrior
from mcda_studio.methods.tradeoff import LOG_BETA_PRIOR_MEAN, LOG_BETA_PRIOR_SD, NoisyFit

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TradeoffProblem(sampler.Problem):
    """The calibration sampler's problem with a Bradley-Terry likelihood of the answers.

    ``contrasts`` is ``(answers, leaves)``: ``v(chosen) - v(other)`` in the prior's leaf order.
    """

    contrasts: Array | None = None

    def log_likelihood(self, global_weights: Array, alpha: Array, beta: Array) -> Array:
        if self.contrasts is None or self.contrasts.shape[0] == 0:
            return np.zeros(alpha.shape[0])
        s = self.contrasts @ global_weights  # (answers, chains)
        total: Array = -np.logaddexp(0.0, -beta[None, :] * s).sum(axis=0)
        return total

    def score_centre(self) -> float:
        return 0.0


def tradeoff_problem(prior: NestedDirichletPrior, contrasts: Array) -> TradeoffProblem:
    """The noisy mode's problem with its stated priors (alpha keeps a standard normal prior)."""
    return TradeoffProblem(
        prior,
        None,
        0.0,
        alpha_sd=1.0,
        log_beta_sd=LOG_BETA_PRIOR_SD,
        log_beta_mean=LOG_BETA_PRIOR_MEAN,
        contrasts=np.atleast_2d(np.asarray(contrasts, dtype=np.float64)),
    )


def fit_noisy(
    prior: NestedDirichletPrior,
    contrasts: Array,
    seed: int,
    settings: sampler.SamplerSettings | None = None,
) -> NoisyFit:
    """Run the sampler (its defaults unless ``settings``) and return thinned draws and the gate."""
    settings = settings or sampler.SamplerSettings(seed=seed)
    settings.check()
    chains = sampler.run(tradeoff_problem(prior, contrasts), settings)
    diagnostics = report.diagnose(prior, chains)
    picks = sampler.thin_indices(settings.chains, settings.draws, settings.samples)
    return NoisyFit(
        draws=np.array([chains.weights[c, i] for c, i in picks]),
        beta=np.array([chains.beta[c, i] for c, i in picks]),
        converged=diagnostics.converged,
        quantities=[(q.name, q.rhat, q.ess_bulk) for q in diagnostics.quantities],
        acceptance=diagnostics.acceptance,
        algorithm=sampler.ALGORITHM,
        chains=settings.chains,
        warmup=settings.warmup,
        draws_per_chain=settings.draws,
        samples=settings.samples,
    )
