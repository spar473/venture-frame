"""``fit``: from a project, a weight profile and a version 2 reference set to posterior weights.

Read-only: the project and the reference set are never changed. Only ``reviewed`` ratings count
unless ``include_proposed`` is set, which marks the run diagnostic. A leaf a case does not rate is
missing, never zero. Design: ``docs/design/calibration.md``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import platform
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from mcda_studio.calibration import report, sampler
from mcda_studio.calibration import weight_samples as ws
from mcda_studio.calibration.likelihood import Observation, OutcomeData, Status, encode
from mcda_studio.calibration.prior import NestedDirichletPrior, make_prior, moment_matched_kappa
from mcda_studio.domain.common import DomainModel, Id
from mcda_studio.domain.errors import McdaError, UnknownReferenceError
from mcda_studio.domain.preferences import WeightProfile
from mcda_studio.domain.project import Project
from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2, date_bounds
from mcda_studio.methods.mavt.value_functions import build_evaluator
from mcda_studio.methods.mavt.weighting import compute_weights
from mcda_studio.services.elicitation_service import CONVICTION_HALF_WIDTH
from mcda_studio.services.reference_case_service import compared_criteria, parse_fraction
from mcda_studio.services.reference_outcome_service import check, window_end

DEFAULT_BASE_RATE = 0.15
MISSING_AT_RANDOM = (
    "Cases whose outcome is unknown or not recorded are left out. This assumes that whether a case "
    "has follow-up does not depend on its outcome given its ratings."
)

Conviction = Literal["high", "medium", "low"]


class FitReport(DomainModel):
    status: Literal["insufficient", "not_converged", "converged"]
    diagnostic: bool
    diagnostic_label: str | None = None
    insufficient: list[str]
    prior_dominated: bool | None = None
    project_id: Id
    project_fingerprint: str
    weight_profile_id: Id
    reference_set_id: Id
    reference_set_sha256: str
    reference_set_digest_basis: str
    data: ws.DataSpec
    prior: ws.PriorSpec | None = None
    seed: int | None = None
    sampler: ws.SamplerSpec | None = None
    diagnostics: report.Diagnostics | None = None
    leaves: list[report.LeafSummary] = []
    parameters: list[report.ParameterSummary] = []
    samples_available: bool = False
    assumption: str = MISSING_AT_RANDOM
    label: str = ws.LABEL
    limitations: list[str] = ws.LIMITATIONS


@dataclass(frozen=True)
class FitOutcome:
    report: FitReport
    samples: ws.WeightSamples | None


@dataclass
class _Cases:
    values: list[list[float]]
    rated: list[list[float]]
    observations: list[Observation]
    statuses: dict[str, int]
    excluded: dict[str, int]
    counted_ratings: int


def _read(reference_set: ReferenceCasesV2 | bytes | str) -> tuple[ReferenceCasesV2, str, str]:
    if isinstance(reference_set, ReferenceCasesV2):
        raw = reference_set.model_dump_json().encode("utf-8")
        return reference_set, hashlib.sha256(raw).hexdigest(), "canonical JSON"
    raw = reference_set.encode("utf-8") if isinstance(reference_set, str) else reference_set
    return (
        ReferenceCasesV2.model_validate_json(raw),
        hashlib.sha256(raw).hexdigest(),
        "file bytes",
    )


def _profile(project: Project, profile_id: str | None) -> WeightProfile:
    if profile_id is None:
        profile = project.resolve_profile(project.analysis)
        if profile is None:
            raise McdaError("the project has no weight profile")
        return profile
    for candidate in project.weight_profiles:
        if candidate.id == profile_id:
            return candidate
    raise UnknownReferenceError(f"unknown weight profile id {profile_id!r}")


def _fraction_of_window(day: dt.date, snapshot: dt.date, length: int) -> float:
    return (day - snapshot).days / length


def _cases(
    project: Project,
    reference_set: ReferenceCasesV2,
    definition: str,
    leaves: list[str],
    include_proposed: bool,
    threshold: Fraction,
) -> _Cases:
    tree = project.tree()
    evaluators = {}
    for leaf in leaves:
        vf = tree.node(leaf).value_function
        if vf is None:
            raise McdaError(f"criterion {leaf!r} has no value function")
        evaluators[leaf] = build_evaluator(vf, None)
    months = reference_set.outcome_definitions[definition].window_months
    out = _Cases([], [], [], {}, {}, 0)
    for case in reference_set.cases:
        ratings = {
            r.criterion_id: r.label
            for r in case.ratings
            if r.criterion_id in evaluators and (r.review == "reviewed" or include_proposed)
        }
        out.counted_ratings += len(ratings)
        found = [o for o in case.outcomes if o.definition == definition]
        reason = None
        if not found:
            reason = "outcome not recorded"
        elif found[0].status == "unknown":
            reason = "outcome unknown"
        elif not ratings or Fraction(len(ratings), len(leaves)) < threshold:
            reason = "too few rated compared criteria"
        if reason is not None:
            out.excluded[reason] = out.excluded.get(reason, 0) + 1
            continue
        outcome = found[0]
        snapshot = case.snapshot_date
        length = (window_end(snapshot, months) - snapshot).days
        status: Status = outcome.status  # type: ignore[assignment]
        if status in ("event", "competing_event"):
            assert outcome.event_date is not None  # guaranteed by validation
            first, last = date_bounds(outcome.event_date)
            if first == last:
                obs = encode(status, time=_fraction_of_window(first, snapshot, length))
            else:
                a = _fraction_of_window(first, snapshot, length)
                b = ((last - snapshot).days + 1) / length
                obs = encode(status, interval=(a, b))
        elif status == "censored":
            assert outcome.observed_through is not None
            obs = encode(
                "censored", time=_fraction_of_window(outcome.observed_through, snapshot, length)
            )
        else:
            obs = encode("no_event")
        out.observations.append(obs)
        out.statuses[status] = out.statuses.get(status, 0) + 1
        out.values.append(
            [evaluators[leaf](ratings[leaf])[0] if leaf in ratings else 0.0 for leaf in leaves]
        )
        out.rated.append([1.0 if leaf in ratings else 0.0 for leaf in leaves])
    return out


def _insufficient(cases: _Cases, include_proposed: bool, threshold: Fraction, n: int) -> list[str]:
    reasons = []
    if cases.counted_ratings == 0:
        reasons.append(
            "0 ratings on the compared criteria"
            if include_proposed
            else "0 reviewed ratings on the compared criteria"
        )
    entering = len(cases.observations)
    events = cases.statuses.get("event", 0)
    if entering == 0:
        reasons.append(
            f"no case with a recorded outcome rates at least {threshold} of the {n} compared "
            "criteria"
        )
    elif events == 0:
        reasons.append("no event among the entering cases")
    elif events == entering:
        reasons.append("every entering case has the event")
    return reasons


def _build_prior(
    project: Project, profile: WeightProfile, half_width: float, local: dict[str, float]
) -> NestedDirichletPrior:
    tree = project.tree()
    held_lists: dict[str | None, list[str]] = {}
    if project.smaa is not None:
        for group in project.smaa.groups:
            held_lists[group.parent_id] = list(group.held)
    children: dict[str | None, list[str]] = {None: tree.child_ids(None)}
    for node in tree.all_ids():
        if not tree.is_leaf(node):
            children[node] = tree.child_ids(node)
    kappa: dict[str | None, float] = {}
    method: dict[str | None, str] = {}
    for parent, kids in children.items():
        share = 1.0 if parent is None else math.prod(local[n] for n in tree.path(parent))
        free = [k for k in kids if local[k] > 0.0 and k not in held_lists.get(parent, [])]
        if share == 0.0 or len(free) < 2:
            continue
        concentration = moment_matched_kappa(
            [float(profile.judgements.get(k, 1.0)) for k in free], half_width
        )
        kappa[parent] = float(concentration.kappa)
        method[parent] = concentration.method
    return make_prior(tree.all_ids(), children, local, kappa, held_lists, method)


def _prior_spec(
    prior: NestedDirichletPrior,
    conviction: Conviction,
    source: str,
    half_width: float,
    base_rate: float,
    problem: sampler.Problem,
) -> ws.PriorSpec:
    return ws.PriorSpec(
        conviction=conviction,
        conviction_source=source,
        half_width=half_width,
        groups=[
            ws.GroupPrior(
                parent_id=g.parent_id,
                children=list(g.children),
                centre=list(g.centre),
                kappa=g.kappa,
                kappa_method=g.kappa_method,
                share=g.share,
                held=list(g.held),
            )
            for g in prior.groups
        ],
        fixed_groups=[
            ws.FixedGroupPrior(parent_id=f.parent_id, children=list(f.children), reason=f.reason)
            for f in prior.fixed
        ],
        base_rate=base_rate,
        alpha_mean=problem.alpha_mean,
        alpha_sd=problem.alpha_sd,
        log_beta_sd=problem.log_beta_sd,
        centre_global=prior.centre_global(),
    )


def fit(
    project: Project,
    profile_id: str | None,
    reference_set: ReferenceCasesV2 | bytes | str,
    *,
    definition: str | None = None,
    base_rate: float = DEFAULT_BASE_RATE,
    conviction: Conviction | None = None,
    include_proposed: bool = False,
    min_rated_fraction: str = "1/2",
    settings: sampler.SamplerSettings | None = None,
) -> FitOutcome:
    """Fit the outcome model and return the report and, when the run converges, the samples."""
    settings = settings or sampler.SamplerSettings()
    settings.check()
    rs, digest, basis = _read(reference_set)
    check(project, rs)
    profile = _profile(project, profile_id)
    tree = project.tree()
    weights = compute_weights(tree, profile.judgements)
    leaves = compared_criteria(tree, weights.global_)
    if not leaves:
        raise McdaError("no labelled criterion carries positive weight in this profile")
    if definition is None:
        if len(rs.outcome_definitions) != 1:
            raise McdaError("the reference set has several outcome definitions; name one")
        definition = next(iter(rs.outcome_definitions))
    elif definition not in rs.outcome_definitions:
        raise UnknownReferenceError(f"unknown outcome definition {definition!r}")
    try:
        threshold = parse_fraction(min_rated_fraction)
    except McdaError:
        threshold = Fraction(0)
    if not 0 < threshold <= 1:
        raise McdaError("the minimum rated fraction must be greater than 0 and at most 1")
    cases = _cases(project, rs, definition, leaves, include_proposed, threshold)
    insufficient = _insufficient(cases, include_proposed, threshold, len(leaves))
    rated_counts = {leaf: int(sum(row[j] for row in cases.rated)) for j, leaf in enumerate(leaves)}
    data_spec = ws.DataSpec(
        outcome_definition=definition,
        min_rated_fraction=str(threshold),
        compared_criteria=leaves,
        entering_cases=len(cases.observations),
        excluded=dict(sorted(cases.excluded.items())),
        statuses=dict(sorted(cases.statuses.items())),
        rated_cases=rated_counts,
        insufficient=insufficient,
    )
    base = FitReport(
        status="insufficient",
        diagnostic=include_proposed,
        diagnostic_label=ws.DIAGNOSTIC_LABEL if include_proposed else None,
        insufficient=insufficient,
        project_id=project.id,
        project_fingerprint=project.fingerprint(),
        weight_profile_id=profile.id,
        reference_set_id=rs.id,
        reference_set_sha256=digest,
        reference_set_digest_basis=basis,
        data=data_spec,
    )
    if insufficient and not include_proposed:
        return FitOutcome(base, None)
    chosen = conviction
    source = "stated at the command line"
    if chosen is None and profile.elicitation is not None:
        chosen = profile.elicitation.conviction
        source = "the profile's elicitation record"
    if chosen is None:
        raise McdaError(
            "no conviction: the profile has no elicitation record with a conviction; state one "
            "(high, medium or low)"
        )
    half_width = CONVICTION_HALF_WIDTH[chosen]
    prior = _build_prior(project, profile, half_width, weights.local)
    data = (
        OutcomeData.build(
            np.array(cases.values),
            np.array(cases.rated),
            cases.observations,
            [prior.leaf_ids.index(leaf) for leaf in leaves],
        )
        if cases.observations
        else None
    )
    problem = sampler.Problem(prior, data, sampler.alpha_prior_mean(base_rate))
    chains = sampler.run(problem, settings)
    diagnostics = report.diagnose(prior, chains)
    leaf_rows = report.leaf_summaries(prior, chains, rated_counts)
    prior_spec = _prior_spec(prior, chosen, source, half_width, base_rate, problem)
    sampler_spec = ws.SamplerSpec(
        algorithm=sampler.ALGORITHM,
        chains=settings.chains,
        warmup=settings.warmup,
        draws=settings.draws,
        samples=settings.samples,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
    )
    result = base.model_copy(
        update={
            "status": "converged" if diagnostics.converged else "not_converged",
            "prior_dominated": report.prior_dominated(leaf_rows),
            "prior": prior_spec,
            "seed": settings.seed,
            "sampler": sampler_spec,
            "diagnostics": diagnostics,
            "leaves": leaf_rows,
            "parameters": report.parameter_summaries(
                chains, problem.alpha_mean, problem.alpha_sd, problem.log_beta_sd
            ),
            "samples_available": diagnostics.converged,
        }
    )
    if not diagnostics.converged:
        return FitOutcome(result, None)
    picks = sampler.thin_indices(settings.chains, settings.draws, settings.samples)
    samples = ws.WeightSamples(
        kind="weight-samples",
        version=1,
        project_id=project.id,
        project_fingerprint=result.project_fingerprint,
        weight_profile_id=profile.id,
        reference_set_id=rs.id,
        reference_set_sha256=digest,
        reference_set_digest_basis=basis,  # type: ignore[arg-type]
        diagnostic=include_proposed,
        usable=not include_proposed and not insufficient,
        review_states=["proposed", "reviewed"] if include_proposed else ["reviewed"],
        data=data_spec,
        prior=prior_spec,
        seed=settings.seed,
        sampler=sampler_spec,
        diagnostics=ws.DiagnosticsSpec.model_validate(diagnostics.model_dump()),
        leaf_ids=list(prior.leaf_ids),
        samples=[[float(x) for x in chains.weights[c, k, :]] for c, k in picks],
        alpha=[float(chains.alpha[c, k]) for c, k in picks],
        beta=[float(chains.beta[c, k]) for c, k in picks],
        label=ws.LABEL if not include_proposed else f"{ws.LABEL} {ws.DIAGNOSTIC_LABEL}",
        limitations=list(ws.LIMITATIONS),
    )
    return FitOutcome(result, samples)
