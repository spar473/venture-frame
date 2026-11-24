"""Robustness bands: which ranks are too close to separate under stated uncertainty.

Design: ``docs/design/robustness-bands.md``. A read-only reading of SMAA-2's rank
acceptabilities. SMAA-2 runs unchanged, through the analysis service, with the project's stated
uncertainty (its ``smaa`` section), the selected profile and scenario, and the saved SMAA-2
parameters (or SMAA-2's defaults). For each alternative it reports the shortest run of
consecutive ranks that holds at least ``COVERAGE`` of the draws, and groups the alternatives
whose runs overlap. Nothing is invented: a project that states no uncertainty, or one SMAA-2
cannot run, gives an unavailable result with its reason.
"""

import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Literal

from mcda_studio.domain.common import DomainModel, Id
from mcda_studio.domain.errors import AnalysisLimitError, ValidationFailed
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.project import Project
from mcda_studio.domain.results import SmaaResult
from mcda_studio.methods.smaa.validation import smaa_parameters
from mcda_studio.services import analysis_service

#: The share of draws a rank range must hold. Held as a fraction so sampled counts compare exactly.
COVERAGE = Fraction(4, 5)
#: Exact enumeration reports float probabilities; their sums may miss COVERAGE by rounding alone.
ENUMERATION_TOLERANCE = 1e-12

LABEL = (
    "Rank ranges show where each alternative's rank falls in most draws under the uncertainty "
    "stated in the project. They are not probabilities of success or predictions."
)
NO_UNCERTAINTY = "No stated uncertainty in this project."

Interval = tuple[int, int]  # 1-based ranks, both ends included


class RobustnessRow(DomainModel):
    alternative_id: Id
    name: str
    mavt_rank: int  # MAVT's competition rank under the same profile and scenario
    rank_low: int
    rank_high: int
    band_share: float  # share of draws whose rank lies in [rank_low, rank_high]
    rank1_share: float  # share of draws at rank 1, tied first places included


class ScreeningRobustness(DomainModel):
    project_fingerprint: str
    configuration: AnalysisConfiguration  # as requested
    weight_profile_id: Id | None  # resolved
    scenario_id: Id | None  # resolved
    available: bool
    reason: str | None = None
    issue_codes: list[str] = []
    coverage: float = float(COVERAGE)
    run: Literal["performance", "preference", "combined"] | None = None
    evaluation: Literal["sampled", "exact_enumeration"] | None = None
    draws: int | None = None  # None under exact enumeration
    seed: int | None = None
    parameters: Literal["saved", "defaults"] | None = None
    rows: list[RobustnessRow] = []  # in MAVT rank order
    groups: list[list[Id]] = []  # overlapping rank ranges; members and groups in MAVT rank order
    basis: str = ""
    label: str = LABEL


def shortest_interval(
    ranks: int, reaches: Callable[[int, int], bool], preferred_rank: int
) -> Interval:
    """The shortest run of consecutive ranks ``[low, high]`` for which ``reaches`` holds.

    Among equally short runs, the one containing ``preferred_rank`` wins, then the lowest ranks.
    ``reaches`` must hold for the full run ``[1, ranks]``.
    """
    for width in range(1, ranks + 1):
        candidates = [
            (low, low + width - 1)
            for low in range(1, ranks - width + 2)
            if reaches(low, low + width - 1)
        ]
        if candidates:
            return min(candidates, key=lambda c: (not c[0] <= preferred_rank <= c[1], c[0]))
    raise ValueError("no run of ranks reaches the coverage")


def rank_range_from_counts(counts: Sequence[int], draws: int, preferred_rank: int) -> Interval:
    """Sampled acceptabilities: exact integer comparison, ``sum * 5 >= 4 * draws``."""

    def reaches(low: int, high: int) -> bool:
        held = sum(counts[low - 1 : high])
        return held * COVERAGE.denominator >= COVERAGE.numerator * draws

    return shortest_interval(len(counts), reaches, preferred_rank)


def rank_range_from_probabilities(probabilities: Sequence[float], preferred_rank: int) -> Interval:
    """Exact enumeration: probabilities summed with ``math.fsum``, within ENUMERATION_TOLERANCE."""

    def reaches(low: int, high: int) -> bool:
        held = math.fsum(probabilities[low - 1 : high])
        return held >= float(COVERAGE) - ENUMERATION_TOLERANCE

    return shortest_interval(len(probabilities), reaches, preferred_rank)


def overlap_groups(ranges: Sequence[tuple[Id, Interval]]) -> list[list[Id]]:
    """Connected components of the interval graph: two ranges overlap when they share a rank.

    ``ranges`` is in MAVT rank order, which each group's members and the groups keep. The sweep
    runs in order of ``low``, never in MAVT order, which could join ranges that do not overlap.
    """
    position = {alt: i for i, (alt, _) in enumerate(ranges)}
    by_low = sorted(ranges, key=lambda item: (item[1][0], item[1][1], position[item[0]]))
    groups: list[list[Id]] = []
    reach = 0
    for alt, (low, high) in by_low:
        if groups and low <= reach:
            groups[-1].append(alt)
            reach = max(reach, high)
        else:
            groups.append([alt])
            reach = high
    ordered = [sorted(group, key=position.__getitem__) for group in groups]
    return sorted(ordered, key=lambda group: position[group[0]])


def _basis(project: Project, result: SmaaResult, saved: bool) -> str:
    """One plain account of what varied and what did not, from the run itself."""
    model = result.uncertainty
    varied: list[str] = []
    if result.run in {"performance", "combined"} and model.cells:
        count = len(model.cells)
        varied.append(f"{count} stated measurement range{'s' if count != 1 else ''}")
    if result.run in {"preference", "combined"}:
        names = {node.id: node.name for node in project.criteria}
        for group in model.groups:
            where = "the top level" if group.parent_id is None else names[group.parent_id]
            if group.kind == "unknown":
                varied.append(
                    f"the weights at {where}, over every split, in place of this profile's"
                )
            else:
                varied.append(
                    f"the swing ratings at {where} within stated intervals, in place of this "
                    "profile's"
                )
    weights = "every weight" if result.run == "performance" or not model.groups else "other weights"
    fixed = [f"{weights} (this profile's)"]
    if result.run == "preference":
        fixed.append("every measurement at its entered value")
    fixed.append("value functions")
    if result.classes:
        fixed.append("class boundaries")
    if result.evaluation == "exact_enumeration":
        how = f"Exact enumeration of {result.outcomes} outcomes"
    else:
        how = f"{result.samples:,} draws, seed {result.seed}"
    source = "saved SMAA-2 parameters" if saved else "SMAA-2 defaults (no saved SMAA-2 parameters)"
    return f"Varied: {'; '.join(varied)}. Fixed: {', '.join(fixed)}. {how}, {source}."


def screening_robustness(
    project: Project, configuration: AnalysisConfiguration | None = None
) -> ScreeningRobustness:
    """Rank ranges and overlap groups for one project revision, profile and scenario. Read-only."""
    requested = (configuration or project.analysis).model_copy(deep=True)
    saved = project.analysis.method == "smaa2"
    parameters = dict(project.analysis.parameters) if saved else {}
    run_config = requested.model_copy(update={"method": "smaa2", "parameters": parameters})

    def unavailable(reason: str, codes: list[str] | None = None) -> ScreeningRobustness:
        return ScreeningRobustness(
            project_fingerprint=project.fingerprint(),
            configuration=requested,
            weight_profile_id=requested.weight_profile_id,
            scenario_id=requested.scenario_id,
            available=False,
            reason=reason,
            issue_codes=codes or [],
        )

    if project.smaa is None:
        return unavailable(NO_UNCERTAINTY)
    try:
        result = analysis_service.evaluate_project(project, run_config)
    except ValidationFailed as exc:
        errors = list(getattr(exc.report, "errors", []))
        first = str(errors[0].message).rstrip(".") if errors else "the model is not valid"
        return unavailable(f"SMAA-2 cannot run: {first}.", [e.code for e in errors])
    except AnalysisLimitError as exc:
        return unavailable(f"SMAA-2 cannot run: {exc}")
    assert isinstance(result, SmaaResult)
    params = smaa_parameters(run_config)
    acceptability = {a.alternative_id: a for a in result.acceptability}
    rows: list[RobustnessRow] = []
    for ranked in result.ranking:
        estimates = acceptability[ranked.alternative_id].ranks
        if result.samples is None:
            low, high = rank_range_from_probabilities([e.frequency for e in estimates], ranked.rank)
            share = math.fsum(e.frequency for e in estimates[low - 1 : high])
        else:
            counts = [e.count or 0 for e in estimates]
            low, high = rank_range_from_counts(counts, result.samples, ranked.rank)
            share = sum(counts[low - 1 : high]) / result.samples
        rows.append(RobustnessRow(
            alternative_id=ranked.alternative_id, name=ranked.name, mavt_rank=ranked.rank,
            rank_low=low, rank_high=high, band_share=share, rank1_share=estimates[0].frequency,
        ))  # fmt: skip
    groups = overlap_groups([(r.alternative_id, (r.rank_low, r.rank_high)) for r in rows])
    return ScreeningRobustness(
        project_fingerprint=result.project_fingerprint,
        configuration=requested,
        weight_profile_id=result.weight_profile_id,
        scenario_id=result.scenario_id,
        available=True,
        run=result.run,
        evaluation=result.evaluation,
        draws=result.samples,
        seed=params.seed,
        parameters="saved" if saved else "defaults",
        rows=rows,
        groups=groups,
        basis=_basis(project, result, saved),
    )
