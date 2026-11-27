"""Revealed preference: weights that fit a log of accept/decline decisions (Layer 2).

``P(accept_i) = sigmoid(alpha + beta * (s_i - S_REF))`` with ``s_i`` the calibration score (the
global weights renormalised over the leaves decision ``i`` rates; missing is never zero) and
``S_REF = 1/2`` the midpoint of the value scale, so ``alpha`` is the log-odds of accepting an
alternative at the reference score. Priors: the nested Dirichlet on the weights, ``alpha ~
Normal(logit p0, 1.5)`` with ``p0`` a stated base accept rate, ``beta ~ LogNormal(log 8, 1)``.
The calibration sampler and its convergence gate are reused unchanged.

Offline and opt-in. Read-only: the project is never changed. Results stay in memory; there is no
stored format for them. Synthetic data only in this version. Design:
``docs/design/revealed-preference.md``.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mcda_studio.calibration import report, sampler
from mcda_studio.calibration.likelihood import scores
from mcda_studio.calibration.service import Conviction, _build_prior, _profile
from mcda_studio.domain.common import DomainModel, Id
from mcda_studio.domain.errors import McdaError
from mcda_studio.domain.project import Project
from mcda_studio.methods.mavt.value_functions import build_evaluator
from mcda_studio.methods.mavt.weighting import compute_weights
from mcda_studio.methods.nested_dirichlet import NestedDirichletPrior
from mcda_studio.services.elicitation_service import CONVICTION_HALF_WIDTH
from mcda_studio.services.reference_case_service import compared_criteria

Array = NDArray[np.float64]

S_REF = 0.5
ALPHA_PRIOR_SD = 1.5
LOG_BETA_PRIOR_MEAN = math.log(8.0)  # beta ~ LogNormal(log 8, 1): median 8
LOG_BETA_PRIOR_SD = 1.0
LINK = "logit"

SYNTHETIC_LABEL = (
    "Synthetic decisions: planted weights and simulated accept/decline calls. This checks the "
    "method; it says nothing about any real decider."
)
LIMITATIONS = [
    "The model is a logistic choice on a linear score; its fit says nothing about alternatives "
    "unlike those in the log.",
    "A log only holds alternatives the decider saw: weights are informed only along the value "
    "contrasts the log contains.",
    "Calls that depend on things outside the criteria act as label noise and attenuate beta.",
    "Weights are identified only where the rated values vary; leaves whose values move together "
    "are identified only in their sum.",
    "Partners are never averaged: one fit per decider unless pooling is stated.",
]


def alpha_prior_mean(base_rate: float) -> float:
    """``logit(p0)``: the prior mean log-odds of accepting an alternative at ``S_REF``."""
    if not 0.0 < base_rate < 1.0:
        raise ValueError("the base accept rate must lie strictly between 0 and 1")
    return math.log(base_rate) - math.log1p(-base_rate)


@dataclass(frozen=True)
class ChoiceData:
    """Decisions: values on the compared leaves and whether each alternative was accepted.

    ``values`` and ``rated`` are ``(decisions, compared leaves)``; a leaf a decision does not rate
    has ``rated = 0`` and its value is ignored. ``accept`` is 1.0 for accept and 0.0 for decline.
    ``leaf_index`` maps each compared leaf to its position in the prior's ``leaf_ids``.
    """

    values: Array
    rated: Array
    accept: Array
    leaf_index: NDArray[np.intp]

    @property
    def cases(self) -> int:
        return int(self.values.shape[0])

    @staticmethod
    def build(values: Array, rated: Array, accept: Array, leaf_index: list[int]) -> ChoiceData:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        rated = np.atleast_2d(np.asarray(rated, dtype=np.float64))
        accept = np.asarray(accept, dtype=np.float64).ravel()
        if values.shape != rated.shape or values.shape[0] != accept.shape[0]:
            raise ValueError("values, rated and accept must describe the same decisions")
        if values.shape[1] != len(leaf_index):
            raise ValueError("every compared leaf needs a position in the prior")
        if not np.all((accept == 0.0) | (accept == 1.0)):
            raise ValueError("a decision is accept (1) or decline (0)")
        if not np.all((rated == 0.0) | (rated == 1.0)):
            raise ValueError("rated is 0 or 1")
        if accept.shape[0] and np.any(rated.sum(axis=1) == 0):
            raise ValueError("every decision rates at least one compared leaf")
        return ChoiceData(
            values=np.where(rated > 0, values, 0.0),
            rated=rated,
            accept=accept,
            leaf_index=np.asarray(leaf_index, dtype=np.intp),
        )


def linear_predictor(data: ChoiceData, global_weights: Array, alpha: Array, beta: Array) -> Array:
    """``eta = alpha + beta * (s - S_REF)``, ``(decisions, chains)``."""
    s = scores(data, global_weights)
    out: Array = alpha[None, :] + beta[None, :] * (s - S_REF)
    return out


def case_log_likelihood(
    data: ChoiceData, global_weights: Array, alpha: Array, beta: Array
) -> Array:
    """Each decision's contribution, ``(decisions, chains)``: ``log sigmoid(+-eta)``, stable."""
    eta = linear_predictor(data, global_weights, alpha, beta)
    y = data.accept[:, None]
    out: Array = -(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta))
    return out


def log_likelihood(
    data: ChoiceData | None, global_weights: Array, alpha: Array, beta: Array
) -> Array:
    """The total log-likelihood per chain, ``(chains,)``; zero without decisions."""
    if data is None or data.cases == 0:
        return np.zeros(alpha.shape[0])
    total: Array = case_log_likelihood(data, global_weights, alpha, beta).sum(axis=0)
    return total


@dataclass(frozen=True)
class ChoiceProblem(sampler.Problem):
    """The calibration sampler's problem with the choice likelihood in place of the outcome one.

    ``data`` (the outcome data) stays ``None``; the decisions are ``choices``.
    """

    choices: ChoiceData | None = None

    def log_likelihood(self, global_weights: Array, alpha: Array, beta: Array) -> Array:
        return log_likelihood(self.choices, global_weights, alpha, beta)

    def score_centre(self) -> float:
        """The mean of ``s_i - S_REF`` at the prior's centre weights (0 without decisions).

        The sampler moves ``a = alpha + beta * centre``; the Jacobian is 1, so the posterior is
        unchanged and only the intercept-slope correlation goes.
        """
        if self.choices is None or self.choices.cases == 0:
            return 0.0
        centre = self.prior.centre_global()
        weights = np.array([[centre[leaf]] for leaf in self.prior.leaf_ids])
        return float(scores(self.choices, weights).mean()) - S_REF


def choice_problem(
    prior: NestedDirichletPrior, choices: ChoiceData | None, base_rate: float
) -> ChoiceProblem:
    """The choice model's problem with its stated priors."""
    return ChoiceProblem(
        prior,
        None,
        alpha_prior_mean(base_rate),
        alpha_sd=ALPHA_PRIOR_SD,
        log_beta_sd=LOG_BETA_PRIOR_SD,
        log_beta_mean=LOG_BETA_PRIOR_MEAN,
        choices=choices,
    )


def parameter_summaries(
    chains: sampler.Chains, problem: ChoiceProblem
) -> list[report.ParameterSummary]:
    return report.parameter_summaries(
        chains, problem.alpha_mean, problem.alpha_sd, problem.log_beta_sd, problem.log_beta_mean
    )


# --- From a project and a list of decisions (read-only) ---------------------------------------


@dataclass(frozen=True)
class Decision:
    """One logged decision: the alternative's labels on the criteria, and the call."""

    labels: dict[str, str]
    accepted: bool
    decided_on: dt.date | None = None
    decider: str | None = None


class ChoiceReport(DomainModel):
    status: str  # "converged" or "not_converged"
    project_id: Id
    project_fingerprint: str
    weight_profile_id: Id
    deciders: list[str]
    pooled: bool
    decisions: int
    accepts: int
    declines: int
    compared_criteria: list[str]
    rated_decisions: dict[str, int]
    base_accept_rate: float
    s_ref: float = S_REF
    link: str = LINK
    conviction: str
    conviction_source: str
    seed: int
    prior_dominated: bool
    diagnostics: report.Diagnostics
    leaves: list[report.LeafSummary]
    parameters: list[report.ParameterSummary]
    limitations: list[str] = LIMITATIONS


@dataclass(frozen=True)
class ChoiceFit:
    """The report and, in memory only, the chains (``None`` when not converged)."""

    report: ChoiceReport
    chains: sampler.Chains | None


def fit_decisions(
    project: Project,
    profile_id: str | None,
    decisions: list[Decision],
    *,
    base_rate: float,
    conviction: Conviction | None = None,
    pool_deciders: bool = False,
    settings: sampler.SamplerSettings | None = None,
) -> ChoiceFit:
    """Fit the choice model to ``decisions`` against ``project``'s value functions and profile.

    ``base_rate`` is required: the stated share accepted at the reference score.
    """
    alpha_prior_mean(base_rate)  # refuses a base rate outside (0, 1) before anything runs
    settings = settings or sampler.SamplerSettings()
    settings.check()
    if not decisions:
        raise McdaError("no decisions")
    deciders = sorted({d.decider for d in decisions if d.decider is not None})
    if len(deciders) > 1 and not pool_deciders:
        raise McdaError(
            f"the log has {len(deciders)} deciders; fit one decider at a time or state a pooling "
            "choice (partners are never averaged)"
        )
    accepts = sum(1 for d in decisions if d.accepted)
    if accepts in (0, len(decisions)):
        raise McdaError("the log needs both accepted and declined decisions")
    profile = _profile(project, profile_id)
    tree = project.tree()
    weights = compute_weights(tree, profile.judgements)
    leaves = compared_criteria(tree, weights.global_)
    if not leaves:
        raise McdaError("no labelled criterion carries positive weight in this profile")
    evaluators = {}
    for leaf in leaves:
        vf = tree.node(leaf).value_function
        if vf is None:
            raise McdaError(f"criterion {leaf!r} has no value function")
        evaluators[leaf] = build_evaluator(vf, None)
    values: list[list[float]] = []
    rated: list[list[float]] = []
    for n, d in enumerate(decisions):
        known = {k: v for k, v in d.labels.items() if k in evaluators}
        if not known:
            raise McdaError(f"decision {n} rates no compared criterion")
        values.append([evaluators[k](known[k])[0] if k in known else 0.0 for k in leaves])
        rated.append([1.0 if k in known else 0.0 for k in leaves])
    chosen = conviction
    source = "stated by the caller"
    if chosen is None and profile.elicitation is not None:
        chosen = profile.elicitation.conviction
        source = "the profile's elicitation record"
    if chosen is None:
        raise McdaError("no conviction: the profile has no elicitation record; state one")
    prior = _build_prior(project, profile, CONVICTION_HALF_WIDTH[chosen], weights.local)
    data = ChoiceData.build(
        np.array(values),
        np.array(rated),
        np.array([1.0 if d.accepted else 0.0 for d in decisions]),
        [prior.leaf_ids.index(leaf) for leaf in leaves],
    )
    problem = choice_problem(prior, data, base_rate)
    chains = sampler.run(problem, settings)
    diagnostics = report.diagnose(prior, chains)
    rated_counts = {leaf: int(data.rated[:, j].sum()) for j, leaf in enumerate(leaves)}
    leaf_rows = report.leaf_summaries(prior, chains, rated_counts)
    result = ChoiceReport(
        status="converged" if diagnostics.converged else "not_converged",
        project_id=project.id,
        project_fingerprint=project.fingerprint(),
        weight_profile_id=profile.id,
        deciders=deciders,
        pooled=pool_deciders and len(deciders) > 1,
        decisions=len(decisions),
        accepts=accepts,
        declines=len(decisions) - accepts,
        compared_criteria=leaves,
        rated_decisions=rated_counts,
        base_accept_rate=base_rate,
        conviction=chosen,
        conviction_source=source,
        seed=settings.seed,
        prior_dominated=report.prior_dominated(leaf_rows),
        diagnostics=diagnostics,
        leaves=leaf_rows,
        parameters=parameter_summaries(chains, problem),
    )
    return ChoiceFit(result, chains if diagnostics.converged else None)
