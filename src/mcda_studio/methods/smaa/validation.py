"""SMAA-2's own parameters and rules for the stated uncertainty (design note §1 and §2)."""

import math
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError

from mcda_studio.domain.common import Issue, Severity
from mcda_studio.domain.criteria import CriterionTree, QualitativeValueFunction, ValueFunctionSpec
from mcda_studio.domain.errors import McdaError
from mcda_studio.domain.measurements import effective_domain
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.project import Project
from mcda_studio.domain.uncertainty import (
    CategoricalDistribution,
    DiscreteDistribution,
    GroupUncertainty,
    TriangularDistribution,
    UniformDistribution,
)
from mcda_studio.methods.smaa import samples
from mcda_studio.methods.sorting.classes import check_sorting

PROBABILITY_TOL = 1e-9
_LEAF_IN_PATH = re.compile(r"\[([^\]]+)\]$")


class SmaaParameters(BaseModel):
    # Unknown keys are reported by validation as UNKNOWN_PARAMETER warnings and ignored here.
    model_config = ConfigDict(extra="ignore", frozen=True)

    #: which stated inputs vary in this run
    uncertainty: Literal["performance", "preference", "combined"] = "combined"
    samples: StrictInt = Field(default=10_000, ge=1_000, le=100_000)
    seed: StrictInt = Field(default=0, ge=0)
    confidence: float = Field(default=0.95, ge=0.8, le=0.999)
    #: where a preference or combined run takes its weights: the stated weight groups, or the
    #: global weight vectors of a supplied weight-samples sidecar (docs/design/
    #: sampled-weights-in-smaa.md); the sidecar's digest pins which file
    weight_source: Literal["stated", "samples"] = "stated"
    weight_samples_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    resample: StrictBool = False


def smaa_parameters(configuration: AnalysisConfiguration) -> SmaaParameters:
    """Parsed parameters; validation has already refused invalid ones."""
    return SmaaParameters.model_validate(configuration.parameters)


def group_children(tree: CriterionTree, parent_id: str | None) -> list[str]:
    return tree.child_ids(parent_id)


def effective_function(
    project: Project, configuration: AnalysisConfiguration, leaf: str
) -> ValueFunctionSpec | None:
    scenario = project.resolve_scenario(configuration)
    node = project.tree().node(leaf)
    if scenario and leaf in scenario.value_function_overrides:
        return scenario.value_function_overrides[leaf]
    return node.value_function


def _probabilities_issue(probabilities: list[float]) -> str | None:
    if not probabilities:
        return "lists no outcomes"
    if any(p < 0 for p in probabilities):
        return "has a negative probability"
    if abs(math.fsum(probabilities) - 1.0) > PROBABILITY_TOL:
        return f"has probabilities summing to {math.fsum(probabilities)!r}, not 1"
    return None


def check_smaa(
    project: Project, configuration: AnalysisConfiguration, value_model_issues: list[Issue]
) -> list[Issue]:
    """The rules for the stated uncertainty, after MAVT's checks of the deterministic model."""
    issues: list[Issue] = []

    def add(code: str, message: str, path: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(Issue(severity=severity, code=code, message=message, path=path))

    try:
        params: SmaaParameters | None = smaa_parameters(configuration)
    except ValidationError:
        params = None  # reported by the parameter checks
    model = project.smaa
    run = params.uncertainty if params else "combined"
    from_samples = samples.uses_samples(params)
    issues.extend(samples.check(project, configuration, params))
    if model is None:
        if not from_samples:
            add(
                "NOTHING_UNCERTAIN",
                "SMAA-2 needs stated uncertainty; the project states none",
                "smaa",
            )
        elif project.sorting is not None:  # class acceptability needs valid classes
            issues.extend(
                i for i in check_sorting(project, []) if i.code != "SORTING_LOCAL_ANCHORS"
            )
        return issues
    if model.dependence != "independent":
        add(
            "UNSUPPORTED_DEPENDENCE",
            f"dependence {model.dependence!r} is not supported: inputs are independent, except "
            "cells listed together in a shared draw",
            "smaa.dependence",
        )
    try:
        tree = project.tree()
        project.resolve_scenario(configuration)
    except McdaError:
        return issues  # the value-model checks report a broken tree or scenario
    alternatives = set(project.alternative_ids())
    leaves = set(tree.leaves())
    local_anchored = {
        m.group(1)
        for i in value_model_issues
        if i.code == "LOCAL_ANCHORS" and (m := _LEAF_IN_PATH.search(i.path))
    }

    seen_cells: set[tuple[str, str]] = set()
    for cell in model.cells:
        key = (cell.alternative_id, cell.criterion_id)
        where = f"smaa.cells[{cell.alternative_id},{cell.criterion_id}]"
        if cell.alternative_id not in alternatives or cell.criterion_id not in leaves:
            add(
                "UNKNOWN_UNCERTAIN_CELL",
                f"{cell.alternative_id!r} on {cell.criterion_id!r} is not an alternative and a "
                "leaf criterion of this project",
                where,
            )
            continue
        if key in seen_cells:
            add("DUPLICATE_UNCERTAIN_CELL", "this cell has more than one distribution", where)
            continue
        seen_cells.add(key)
        if cell.criterion_id in local_anchored:
            add(
                "SMAA_LOCAL_ANCHORS_UNCERTAIN",
                f"{cell.criterion_id!r} uses local anchors, which would move with every draw; "
                "use fixed anchors for a criterion with uncertain measurements",
                where,
            )
            continue
        node = tree.node(cell.criterion_id)
        vf = effective_function(project, configuration, cell.criterion_id)
        if node.scale is None or vf is None:
            continue  # the value-model checks report the missing scale or function
        dist = cell.distribution
        labelled = isinstance(vf, QualitativeValueFunction)
        if labelled != isinstance(dist, CategoricalDistribution):
            add(
                "DISTRIBUTION_SCALE_MISMATCH",
                "a labelled scale needs a categorical distribution, and a numeric scale a "
                "uniform, triangular or discrete one",
                where,
            )
            continue
        problem: str | None = None
        support: list[float] = []
        if isinstance(dist, UniformDistribution):
            if not dist.lower < dist.upper:
                problem = "needs lower < upper"
            support = [dist.lower, dist.upper]
        elif isinstance(dist, TriangularDistribution):
            if not (dist.lower <= dist.mode <= dist.upper and dist.lower < dist.upper):
                problem = "needs lower <= mode <= upper and lower < upper"
            support = [dist.lower, dist.upper]
        elif isinstance(dist, DiscreteDistribution):
            if len(dist.values) != len(dist.probabilities):
                problem = "needs one probability per value"
            elif len(set(dist.values)) != len(dist.values):
                problem = "lists a value twice"
            else:
                problem = _probabilities_issue(dist.probabilities)
            support = list(dist.values)
        else:
            if len(dist.labels) != len(dist.probabilities):
                problem = "needs one probability per label"
            elif len(set(dist.labels)) != len(dist.labels):
                problem = "lists a label twice"
            elif unknown := [x for x in dist.labels if x not in node.scale.labels]:
                problem = f"lists labels that are not on the scale: {unknown}"
            else:
                problem = _probabilities_issue(dist.probabilities)
        if problem:
            add("INVALID_DISTRIBUTION", f"the {dist.kind} distribution {problem}", where)
            continue
        domain = effective_domain(node.scale, vf, [])
        if (
            support
            and domain is not None
            and (min(support) < domain.lower or max(support) > domain.upper)
        ):
            add(
                "DISTRIBUTION_OUTSIDE_DOMAIN",
                f"the distribution reaches outside the value-function domain "
                f"[{domain.lower}, {domain.upper}]; it is never clamped",
                where,
            )

    parents = {None, *(i for i in tree.all_ids() if not tree.is_leaf(i))}
    seen_groups: set[str | None] = set()
    for group in model.groups:
        label = "top" if group.parent_id is None else group.parent_id
        where = f"smaa.groups[{label}]"
        if group.parent_id not in parents:
            add(
                "UNKNOWN_UNCERTAIN_GROUP",
                f"{group.parent_id!r} is not a criterion with children",
                where,
            )
            continue
        if group.parent_id in seen_groups:
            add("DUPLICATE_UNCERTAIN_GROUP", "this group has more than one entry", where)
            continue
        seen_groups.add(group.parent_id)
        children = group_children(tree, group.parent_id)
        if len(children) < 2:
            add(
                "INVALID_GROUP_UNCERTAINTY",
                "a group with one child always has local weight 1, so it cannot vary",
                where,
            )
            continue
        held = _held_children(group, children, where, add)
        varying = [c for c in children if c not in held]
        if len(varying) < 2:
            add(
                "INVALID_GROUP_UNCERTAINTY",
                "fewer than two children vary once the held children are kept at the profile's "
                "weight",
                f"{where}.held",
            )
            continue
        if held:
            _check_held_share(project, configuration, children, held, where, add)
        if group.kind != "ranking" and group.order:
            add(
                "UNEXPECTED_ORDER",
                f"an {group.kind} group takes no order; only a ranking group does",
                f"{where}.order",
            )
        if group.kind in {"unknown", "ranking"}:
            if group.intervals:
                add(
                    "UNEXPECTED_RATING_INTERVALS",
                    f"an {group.kind} group draws local weights; it takes no rating intervals",
                    where,
                )
            if group.kind == "ranking":
                _check_order(group, varying, held, where, add)
            continue
        for child in varying:
            interval = group.intervals.get(child)
            if interval is None:
                add("MISSING_RATING_INTERVAL", f"no rating interval for {child!r}", where)
            elif not 0.0 <= interval[0] <= interval[1]:
                add(
                    "INVALID_RATING_INTERVAL",
                    f"the interval for {child!r} needs 0 <= lower <= upper",
                    f"{where}.intervals.{child}",
                )
        for extra in sorted(set(group.intervals) - set(children)):
            add(
                "UNKNOWN_RATING_INTERVAL",
                f"{extra!r} is not a child of this group",
                f"{where}.intervals.{extra}",
            )
        for child in sorted(set(group.intervals) & set(held)):
            add(
                "UNEXPECTED_RATING_INTERVALS",
                f"{child!r} is held at the profile's weight; it takes no rating interval",
                f"{where}.intervals.{child}",
            )
        stated = [group.intervals[c] for c in varying if c in group.intervals]
        if stated and all(hi == 0.0 for _, hi in stated):
            add(
                "ZERO_RATING_INTERVALS",
                "every rating interval is zero, so every draw of this group sums to zero",
                where,
            )

    shared_ids: set[str] = set()
    in_shared: set[tuple[str, str]] = set()
    for shared in model.shared_draws:
        where = f"smaa.shared_draws[{shared.id}]"
        if shared.id in shared_ids:
            add("DUPLICATE_SHARED_DRAW", "this shared draw id is used twice", where)
        shared_ids.add(shared.id)
        if len(shared.cells) < 2:
            add("INVALID_SHARED_DRAW", "a shared draw needs at least two cells", where)
        for ref in shared.cells:
            key = (ref.alternative_id, ref.criterion_id)
            if key not in seen_cells:
                add(
                    "UNKNOWN_SHARED_CELL",
                    f"{ref.alternative_id!r} on {ref.criterion_id!r} has no stated distribution",
                    where,
                )
            elif key in in_shared:
                add("DUPLICATE_SHARED_CELL", "a cell can be in one shared draw only", where)
            in_shared.add(key)

    varies_cells = run in {"performance", "combined"} and bool(model.cells)
    varies_groups = run in {"preference", "combined"} and bool(model.groups)
    if not (varies_cells or varies_groups or from_samples):
        add(
            "NOTHING_UNCERTAIN",
            f"nothing stated varies in a {run} run",
            "configuration.parameters.uncertainty",
        )
    if project.sorting is not None:  # class acceptability needs valid classes
        issues.extend(i for i in check_sorting(project, []) if i.code != "SORTING_LOCAL_ANCHORS")
    return issues


def _held_children(
    group: GroupUncertainty, children: list[str], where: str, add: Callable[[str, str, str], None]
) -> list[str]:
    """The held children that are children of this group, each once; the rest are refused."""
    seen: list[str] = []
    for child in group.held:
        if child not in children:
            add("UNKNOWN_HELD_CHILD", f"{child!r} is not a child of this group", f"{where}.held")
        elif child in seen:
            add("DUPLICATE_HELD_CHILD", f"{child!r} is held twice", f"{where}.held")
        else:
            seen.append(child)
    return seen


def _check_order(
    group: GroupUncertainty,
    varying: list[str],
    held: list[str],
    where: str,
    add: Callable[[str, str, str], None],
) -> None:
    """A ranking names every varying child exactly once, best first, and no held child."""
    path = f"{where}.order"
    if not group.order:
        add("MISSING_ORDER", "a ranking group needs its children in order, best first", path)
        return
    for child in group.order:
        if child in held:
            add("INVALID_ORDER", f"{child!r} is held, so it takes no place in the order", path)
        elif child not in varying:
            add("INVALID_ORDER", f"{child!r} is not a child of this group", path)
    repeated = sorted({c for c in group.order if group.order.count(c) > 1})
    if repeated:
        add(
            "INVALID_ORDER",
            f"the order names {repeated} more than once; ties are not allowed",
            path,
        )
    missing = [c for c in varying if c not in group.order]
    if missing:
        add("INVALID_ORDER", f"the order leaves out {missing}", path)


def _check_held_share(
    project: Project,
    configuration: AnalysisConfiguration,
    children: list[str],
    held: list[str],
    where: str,
    add: Callable[[str, str, str], None],
) -> None:
    """Held children keep the profile's local weight, so they must leave some weight to share."""
    try:
        profile = project.resolve_profile(configuration)
    except McdaError:
        return  # MAVT's own checks already refuse an unknown profile
    if profile is None:
        return  # MAVT's own checks already refuse a missing profile
    ratings = [profile.judgements.get(c) for c in children]
    if any(r is None for r in ratings):
        return  # MAVT's own checks already refuse an incomplete group
    total = math.fsum(r for r in ratings if r is not None)
    kept = math.fsum(profile.judgements[c] for c in held)
    if total > 0.0 and kept >= total:
        add(
            "HELD_TAKES_ALL_WEIGHT",
            "the held children have all of this group's weight in the profile, so nothing is "
            "left to vary",
            f"{where}.held",
        )
