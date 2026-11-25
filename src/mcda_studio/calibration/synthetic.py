"""A synthetic bench: planted weights, simulated outcomes, and a recovery check.

The bench tree has two objectives and five leaves, each rated on a five-label scale with values
0, 1/4, 1/2, 3/4 and 1. Outcomes follow the model in ``likelihood``: exponential event times with
hazard ``exp(alpha + beta * s)`` per window, optional competing events, a follow-up mechanism that
censors, partial event dates, missing ratings and cases with no follow-up (missing at random).

``censoring="score"`` gives shorter follow-up to higher scores (later snapshots, say). Censoring is
then informative about the score but not about the outcome given the score, so the
censoring-aware likelihood stays unbiased while the naive treatments do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mcda_studio.calibration import report
from mcda_studio.calibration.likelihood import Observation, OutcomeData, encode
from mcda_studio.calibration.prior import NestedDirichletPrior, make_prior
from mcda_studio.calibration.sampler import Problem, SamplerSettings, alpha_prior_mean, run
from mcda_studio.domain.common import DomainModel

Array = NDArray[np.float64]
Censoring = Literal["none", "independent", "score"]

NODES = ("obj_a", "a1", "a2", "a3", "obj_b", "b1", "b2")
CHILDREN: dict[str | None, tuple[str, ...]] = {
    None: ("obj_a", "obj_b"),
    "obj_a": ("a1", "a2", "a3"),
    "obj_b": ("b1", "b2"),
}
LEAVES = ("a1", "a2", "a3", "b1", "b2")
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: The stated profile the prior is centred on.
STATED = {"obj_a": 0.6, "obj_b": 0.4, "a1": 0.5, "a2": 0.3, "a3": 0.2, "b1": 0.7, "b2": 0.3}
#: Planted weights for the recovery check, deliberately away from the stated ones.
PLANTED = {"obj_a": 0.45, "obj_b": 0.55, "a1": 0.2, "a2": 0.5, "a3": 0.3, "b1": 0.35, "b2": 0.65}
PLANTED_ALPHA = -2.0
PLANTED_BETA = 3.0
BASE_RATE = 0.15
PARTIAL_PARTS = 12  # a partial event date covers one twelfth of the window


@dataclass(frozen=True)
class Truth:
    local: dict[str, float]
    alpha: float
    beta: float

    def global_weights(self, prior: NestedDirichletPrior) -> dict[str, float]:
        return {
            leaf: math.prod(self.local[n] for n in prior.paths[leaf]) for leaf in prior.leaf_ids
        }


@dataclass(frozen=True)
class SyntheticSet:
    data: OutcomeData
    statuses: list[str]
    observations: list[Observation]
    truth: Truth
    excluded_unknown: int


def bench_prior(kappa: float | dict[str | None, float] = 47.0) -> NestedDirichletPrior:
    kappas = kappa if isinstance(kappa, dict) else {p: float(kappa) for p in CHILDREN}
    return make_prior(NODES, CHILDREN, STATED, kappas)


def planted() -> Truth:
    return Truth(dict(PLANTED), PLANTED_ALPHA, PLANTED_BETA)


def draw_truth(
    prior: NestedDirichletPrior, rng: np.random.Generator, alpha_mean: float, alpha_sd: float = 1.5
) -> Truth:
    """A truth drawn from the prior (for coverage and simulation-based calibration)."""
    ys = [y[:1] for y in prior.sample_ys(rng, 1)]
    local = prior.local(ys, 1)[:-1, 0]
    alpha = alpha_mean + alpha_sd * float(rng.standard_normal())
    beta = math.exp(float(rng.standard_normal()))
    return Truth(dict(zip(prior.node_ids, (float(v) for v in local), strict=True)), alpha, beta)


def simulate(
    prior: NestedDirichletPrior,
    truth: Truth,
    cases: int,
    rng: np.random.Generator,
    *,
    censoring: Censoring = "independent",
    censored_share: float = 0.3,
    censor_scale: float = 2.2,
    competing_rate: float = 0.0,
    partial_share: float = 0.0,
    missing_share: float = 0.0,
    unknown_share: float = 0.0,
) -> SyntheticSet:
    """Simulate ``cases`` cases; those with no follow-up or too few ratings do not enter."""
    leaves = list(prior.leaf_ids)
    k = len(leaves)
    values = rng.choice(np.array(LEVELS), size=(cases, k))
    rated = (rng.random((cases, k)) >= missing_share).astype(float)
    weights = truth.global_weights(prior)
    w = np.array([weights[leaf] for leaf in leaves])
    s = (rated * values * w).sum(axis=1) / np.maximum((rated * w).sum(axis=1), 1e-300)
    lam = np.exp(truth.alpha + truth.beta * s)
    event_time = rng.exponential(1.0 / lam)
    competing_time = (
        rng.exponential(1.0 / competing_rate, size=cases)
        if competing_rate > 0
        else np.full(cases, np.inf)
    )
    u = rng.random(cases)
    v = rng.uniform(0.25, 1.25, size=cases)
    if censoring == "none":
        follow_up = np.full(cases, np.inf)
    elif censoring == "independent":
        follow_up = np.where(u < censored_share, rng.random(cases), np.inf)
    else:
        follow_up = censor_scale * (1.0 - s) * v
    partial = rng.random(cases) < partial_share
    unknown = rng.random(cases) < unknown_share
    keep = (2 * rated.sum(axis=1) >= k) & (rated.sum(axis=1) > 0)
    statuses: list[str] = []
    observations: list[Observation] = []
    rows: list[int] = []
    for i in range(cases):
        if unknown[i] or not keep[i]:
            continue
        t, c, f = event_time[i], competing_time[i], follow_up[i]
        if t <= min(c, f, 1.0):
            status = "event"
            if partial[i]:
                a = math.floor(t * PARTIAL_PARTS) / PARTIAL_PARTS
                obs = encode("event", interval=(a, min(1.0, a + 1.0 / PARTIAL_PARTS)))
            else:
                obs = encode("event", time=float(t))
        elif c < t and c <= min(f, 1.0):
            status, obs = "competing_event", encode("competing_event", time=float(c))
        elif f >= 1.0:
            status, obs = "no_event", encode("no_event")
        else:
            status, obs = "censored", encode("censored", time=float(f))
        statuses.append(status)
        observations.append(obs)
        rows.append(i)
    data = OutcomeData.build(values[rows], rated[rows], observations, list(range(k)))
    return SyntheticSet(
        data=data,
        statuses=statuses,
        observations=observations,
        truth=truth,
        excluded_unknown=int(unknown.sum()),
    )


def censoring_variant(sset: SyntheticSet, how: Literal["drop", "no_event"]) -> OutcomeData:
    """A naive treatment of censored cases, for comparison only: drop them, or call them no event.

    The service never does either; this exists to show what the censoring-aware likelihood avoids.
    """
    keep = [i for i, s in enumerate(sset.statuses) if not (how == "drop" and s == "censored")]
    observations = [
        encode("no_event")
        if how == "no_event" and sset.statuses[i] == "censored"
        else sset.observations[i]
        for i in keep
    ]
    return OutcomeData.build(
        sset.data.values[keep], sset.data.rated[keep], observations, list(sset.data.leaf_index)
    )


class TruthRow(DomainModel):
    name: str
    truth: float
    posterior_mean: float
    interval: tuple[float, float]
    covered: bool


class SyntheticCheck(DomainModel):
    cases_simulated: int
    cases_entering: int
    counts: dict[str, int]
    censoring: str
    seed: int
    rows: list[TruthRow]
    diagnostics: report.Diagnostics
    label: str = (
        "Synthetic data: planted weights and simulated outcomes. This checks the method; it says "
        "nothing about any real outcome."
    )


def synthetic_check(
    cases: int = 500,
    seed: int = 0,
    censoring: Censoring = "independent",
    settings: SamplerSettings | None = None,
) -> SyntheticCheck:
    """Plant weights, simulate, fit, and report which 90% intervals cover the truth."""
    prior = bench_prior()
    truth = planted()
    sset = simulate(
        prior, truth, cases, np.random.Generator(np.random.PCG64(seed)), censoring=censoring
    )
    settings = settings or SamplerSettings(seed=seed)
    problem = Problem(prior, sset.data, alpha_prior_mean(BASE_RATE))
    chains = run(problem, settings)
    leaves = report.leaf_summaries(prior, chains)
    params = report.parameter_summaries(
        chains, problem.alpha_mean, problem.alpha_sd, problem.log_beta_sd
    )
    planted_global = truth.global_weights(prior)
    rows = [
        TruthRow(
            name=f"w[{leaf.criterion_id}]",
            truth=planted_global[leaf.criterion_id],
            posterior_mean=leaf.posterior_mean,
            interval=leaf.interval,
            covered=leaf.interval[0] <= planted_global[leaf.criterion_id] <= leaf.interval[1],
        )
        for leaf in leaves
    ] + [
        TruthRow(
            name=p.name,
            truth=truth.alpha if p.name == "alpha" else truth.beta,
            posterior_mean=p.posterior_mean,
            interval=p.interval,
            covered=p.interval[0]
            <= (truth.alpha if p.name == "alpha" else truth.beta)
            <= p.interval[1],
        )
        for p in params
    ]
    counts: dict[str, int] = {}
    for s in sset.statuses:
        counts[s] = counts.get(s, 0) + 1
    return SyntheticCheck(
        cases_simulated=cases,
        cases_entering=sset.data.cases,
        counts=counts,
        censoring=censoring,
        seed=seed,
        rows=rows,
        diagnostics=report.diagnose(prior, chains),
    )
