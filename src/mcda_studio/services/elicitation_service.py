"""Structured weight judgement: checks of an elicitation record, and the weight uncertainty it
proposes.

Read-only. A profile's swing ratings stay the judgement of record; the elicitation record adds the
order the person gave, optional AHP answers as a cross-check, a conviction level and a reason per
weight. This service derives, and never stores:

- for every sibling group: the swing local weights, rank-order centroid (ROC) weights for the
  stated order, AHP priorities and consistency ratio, whether the order and the ratings agree, the
  largest gap between swing and AHP weights, and which reasons are missing;
- two proposals for SMAA-2's weight uncertainty: **conviction** (each nonzero swing rating within
  a stated band around itself) and **ranking** (every split that keeps the stated order). A
  child the profile rates 0, such as a screened-only mandate, is held at 0 in both.

A proposal is only a proposal: it changes the project when a person applies it.
Design: ``docs/design/structured-judgement.md``.
"""

from __future__ import annotations

import math
from typing import Literal

from mcda_studio.domain.common import DomainModel, Id
from mcda_studio.domain.errors import McdaError, UnknownReferenceError
from mcda_studio.domain.preferences import AnalysisConfiguration, GroupElicitation, WeightProfile
from mcda_studio.domain.project import Project
from mcda_studio.domain.uncertainty import GroupUncertainty, SmaaModel
from mcda_studio.methods.ahp.matrix import CR_THRESHOLD, analyse, matrix_issues
from mcda_studio.services import screening_robustness_service as robustness

#: Conviction becomes rating intervals: each nonzero swing rating r is drawn uniformly from
#: [r(1 - h), r(1 + h)]. The half-widths are a stated convention, shown with every proposal, not
#: an estimate; the design note gives their spread in Dirichlet-equivalent terms.
CONVICTION_HALF_WIDTH: dict[str, float] = {"high": 0.15, "medium": 0.30, "low": 0.50}
#: A swing and an AHP local weight further apart than this ask for a second look.
GAP_THRESHOLD = 0.10
LABEL = (
    "Checks and proposals are derived from the profile's swing ratings and its elicitation "
    "record; nothing here changes the project until a person applies a proposal."
)

Status = Literal["consistent", "check", "incomplete"]


class ChildCheck(DomainModel):
    criterion_id: Id
    name: str
    rating: float | None  # the profile's swing rating; None when not supplied
    swing: float | None  # local weight from the swing ratings
    roc: float | None  # rank-order centroid weight for the stated order, on the same footing
    ahp: float | None  # AHP priority from the pairwise answers, on the same footing
    held: bool  # rated 0 by the profile: held at 0 in every proposal
    reason: str


class AhpCheck(DomainModel):
    complete: bool
    consistency_ratio: float | None = None
    inconsistent: bool = False  # consistency ratio above the conventional 0.10
    triad: list[Id] = []  # the most inconsistent triple, when there are three or more items
    issues: list[str] = []


class GroupCheck(DomainModel):
    parent_id: Id | None
    name: str
    scored: bool  # False under a criterion rated 0 (a screened-only branch): never varied
    children: list[ChildCheck]
    order: list[Id]
    # The children rated above 0, by swing rating (highest first, ties in tree order): a starting
    # point for a ranking the person then confirms or changes. Empty without complete ratings.
    suggested_order: list[Id]
    order_agrees: bool | None  # None without an order or complete ratings
    order_conflicts: list[tuple[Id, Id]]  # (ranked higher, ranked lower) with a lower rating
    ahp: AhpCheck | None  # None without pairwise answers
    largest_gap: float | None  # largest |swing - AHP| local weight
    missing_reasons: list[Id]
    findings: list[str]
    status: Status


class Proposal(DomainModel):
    mode: Literal["conviction", "ranking"]
    label: str
    explanation: str
    smaa: SmaaModel  # the project's stated measurement ranges with these weight groups


class ElicitationCheck(DomainModel):
    project_fingerprint: str
    profile_id: Id
    profile_name: str
    recorded: bool
    elicited_by: str = ""
    elicited_on: str | None = None
    conviction: str | None = None
    groups: list[GroupCheck] = []
    proposals: list[Proposal] = []
    status: Status
    label: str = LABEL


class ProposalPreview(DomainModel):
    mode: Literal["current", "conviction", "ranking"]
    label: str
    robustness: robustness.ScreeningRobustness


class ElicitationPreview(DomainModel):
    project_fingerprint: str
    profile_id: Id
    previews: list[ProposalPreview]
    label: str = (
        "Decision confidence under the project's current stated uncertainty and under each "
        "proposal, from the same SMAA-2 settings. Read-only."
    )


# ---- pure helpers (also used by the tests) ------------------------------------------------------
def roc_weights(n: int) -> list[float]:
    """Rank-order centroid weights (Barron and Barrett, 1996) for ranks 1..n, best first:
    w_k = (1/n) * sum_{i=k}^{n} 1/i. They are the centre of every split that keeps the order."""
    return [math.fsum(1.0 / i for i in range(k, n + 1)) / n for k in range(1, n + 1)]


def order_conflicts(order: list[str], ratings: dict[str, float]) -> list[tuple[str, str]]:
    """Every pair in which the child ranked higher has a strictly lower swing rating."""
    return [
        (hi, lo) for i, hi in enumerate(order) for lo in order[i + 1 :] if ratings[hi] < ratings[lo]
    ]


def conviction_intervals(
    ratings: dict[str, float], half_width: float
) -> dict[str, tuple[float, float]]:
    return {c: (r * (1.0 - half_width), r * (1.0 + half_width)) for c, r in ratings.items()}


# ---- the check ---------------------------------------------------------------------------------
def _resolve(project: Project, profile_id: str | None) -> WeightProfile:
    if profile_id is None:
        if not project.weight_profiles:
            raise McdaError("the project has no weight profile")
        return project.weight_profiles[0]
    for profile in project.weight_profiles:
        if profile.id == profile_id:
            return profile
    raise UnknownReferenceError(f"unknown weight profile id {profile_id!r}")


def _groups(project: Project) -> list[tuple[str | None, list[str]]]:
    tree = project.tree()
    parents: list[str | None] = [None, *(i for i in tree.all_ids() if not tree.is_leaf(i))]
    return [(p, tree.child_ids(p)) for p in parents if len(tree.child_ids(p)) >= 2]


def _group_check(
    project: Project, parent: str | None, children: list[str], profile: WeightProfile
) -> GroupCheck:
    names = {n.id: n.name for n in project.criteria}
    name = "the top level" if parent is None else names[parent]
    record = next(
        (g for g in (profile.elicitation.groups if profile.elicitation else []) if
         g.parent_id == parent),
        GroupElicitation(parent_id=parent),
    )  # fmt: skip
    ratings = {c: profile.judgements.get(c) for c in children}
    complete = all(r is not None for r in ratings.values())
    total = math.fsum(r for r in ratings.values() if r is not None)
    rated = {c: r for c, r in ratings.items() if r is not None}
    swing = {c: r / total for c, r in rated.items()} if complete and total > 0 else {}
    held = [c for c, r in rated.items() if r == 0.0]
    ranked = [c for c in children if c not in held]
    findings: list[str] = []

    # order
    order = list(record.order)
    order_ok = sorted(order) == sorted(ranked) and len(set(order)) == len(order)
    if order and not order_ok:
        expected = ", ".join(names[c] for c in ranked)
        findings.append(
            f"The order should name every child rated above 0 exactly once ({expected})."
        )
    roc = dict(zip(order, roc_weights(len(order)), strict=True)) if order_ok else {}
    conflicts: list[tuple[str, str]] = []
    agrees: bool | None = None
    if order_ok and complete:
        conflicts = order_conflicts(order, {c: rated[c] for c in order})
        agrees = not conflicts
        for hi, lo in conflicts:
            findings.append(
                f"The order puts {names[hi]} above {names[lo]}, but the swing ratings give "
                f"{names[lo]} more ({rated[lo]:g} against {rated[hi]:g})."
            )

    # AHP cross-check over the ranked children
    ahp_check: AhpCheck | None = None
    ahp: dict[str, float] = {}
    if record.pairwise:
        issues = matrix_issues(ranked, record.pairwise, f"elicitation.groups[{parent}]")
        if issues:
            ahp_check = AhpCheck(complete=False, issues=[i.message for i in issues])
            findings.append("The pairwise answers are incomplete or invalid, so no AHP check.")
        else:
            analysis = analyse(ranked, record.pairwise)
            ahp = dict(zip(ranked, analysis.priorities, strict=True))
            ahp_check = AhpCheck(
                complete=True,
                consistency_ratio=analysis.consistency_ratio,
                inconsistent=analysis.inconsistent,
                triad=list(analysis.triad.items) if analysis.triad else [],
            )
            if analysis.inconsistent:
                findings.append(
                    f"The pairwise answers are inconsistent (consistency ratio "
                    f"{analysis.consistency_ratio:.2f}, above {CR_THRESHOLD:.2f})."
                )

    # Held children are rated 0, so the ranked children carry all the swing weight: ROC and AHP
    # weights over the ranked children are already on the swing weights' footing.
    gap: float | None = None
    if ahp and swing:
        gap = max(abs(swing[c] - ahp[c]) for c in ranked)
        if gap > GAP_THRESHOLD:
            worst = max(ranked, key=lambda c: abs(swing[c] - ahp[c]))
            findings.append(
                f"Swing and AHP weights differ by up to {gap:.2f} (largest for {names[worst]}); "
                "the two ways of asking disagree."
            )
    missing = [c for c in ranked if not record.reasons.get(c, "").strip()]
    if not complete:
        findings.insert(0, "Some swing ratings are missing, so the weights cannot be checked.")
    rows = [
        ChildCheck(
            criterion_id=c,
            name=names[c],
            rating=ratings[c],
            swing=swing.get(c),
            roc=roc.get(c, 0.0 if c in held and roc else None),
            ahp=ahp.get(c, 0.0 if c in held and ahp else None),
            held=c in held,
            reason=record.reasons.get(c, ""),
        )
        for c in children
    ]
    status: Status
    if not complete:
        status = "incomplete"
    elif conflicts or (ahp_check and (ahp_check.inconsistent or not ahp_check.complete)) or (
        gap is not None and gap > GAP_THRESHOLD) or (order and not order_ok):  # fmt: skip
        status = "check"
    else:
        status = "consistent"
    return GroupCheck(
        parent_id=parent,
        name=name,
        scored=_scored(project, profile, parent),
        children=rows,
        order=order,
        suggested_order=sorted(ranked, key=lambda c: -rated[c]) if complete else [],
        order_agrees=agrees,
        order_conflicts=conflicts,
        ahp=ahp_check,
        largest_gap=gap,
        missing_reasons=missing,
        findings=findings,
        status=status,
    )


def _scored(project: Project, profile: WeightProfile, parent: str | None) -> bool:
    """Whether a group's weights can matter: no criterion on its path is rated 0 among its
    siblings (a screened-only branch such as a mandate carries no weight, so it is not varied)."""
    tree = project.tree()
    node = parent
    while node is not None:
        if profile.judgements.get(node) == 0.0:
            return False
        node = tree.node(node).parent_id
    return True


def _proposals(
    project: Project, profile: WeightProfile, checks: list[GroupCheck]
) -> list[Proposal]:
    record = profile.elicitation
    base = project.smaa or SmaaModel()
    out: list[Proposal] = []
    ready = [g for g in checks if g.status != "incomplete" and g.scored]
    conviction = record.conviction if record else None
    if conviction is not None:
        h = CONVICTION_HALF_WIDTH[conviction]
        groups = []
        for g in ready:
            nonzero = {c.criterion_id: c.rating for c in g.children if c.rating and not c.held}
            if len(nonzero) < 2:
                continue
            groups.append(
                GroupUncertainty(
                    parent_id=g.parent_id,
                    kind="rating_intervals",
                    intervals=conviction_intervals(nonzero, h),
                    held=[c.criterion_id for c in g.children if c.held],
                )
            )
        if groups:
            out.append(
                Proposal(
                    mode="conviction",
                    label=f"{conviction.capitalize()} conviction: every swing rating ±{h:.0%}",
                    explanation=(
                        f"Each swing rating above 0 is drawn uniformly within ±{h:.0%} of itself "
                        f"({conviction} conviction), in every scored sibling group; children "
                        "rated 0 are held at 0, and groups under them are not varied."
                    ),
                    smaa=base.model_copy(update={"groups": groups}, deep=True),
                )
            )
    ranked_groups = [
        GroupUncertainty(
            parent_id=g.parent_id,
            kind="ranking",
            order=g.order,
            held=[c.criterion_id for c in g.children if c.held],
        )
        for g in ready
        if g.order and len(g.order) >= 2 and not any("order should name" in f for f in g.findings)
    ]
    if ranked_groups:
        out.append(
            Proposal(
                mode="ranking",
                label="Ranking only: every split that keeps the stated order",
                explanation=(
                    "In each ranked group, every split of the weights that keeps the stated order "
                    "is equally likely; the swing ratings' sizes are set aside. Children rated 0 "
                    "are held at 0; groups without an order keep the profile's ratings."
                ),
                smaa=base.model_copy(update={"groups": ranked_groups}, deep=True),
            )
        )
    return out


def elicitation_check(project: Project, profile_id: str | None = None) -> ElicitationCheck:
    profile = _resolve(project, profile_id)
    checks = [_group_check(project, p, kids, profile) for p, kids in _groups(project)]
    record = profile.elicitation
    statuses = {g.status for g in checks}
    overall: Status = (
        "incomplete" if "incomplete" in statuses else "check" if "check" in statuses else
        "consistent"
    )  # fmt: skip
    return ElicitationCheck(
        project_fingerprint=project.fingerprint(),
        profile_id=profile.id,
        profile_name=profile.name,
        recorded=record is not None,
        elicited_by=record.elicited_by if record else "",
        elicited_on=record.elicited_on.isoformat() if record and record.elicited_on else None,
        conviction=record.conviction if record else None,
        groups=checks,
        proposals=_proposals(project, profile, checks),
        status=overall,
    )


def elicitation_preview(
    project: Project, configuration: AnalysisConfiguration
) -> ElicitationPreview:
    """Decision confidence now and under each proposal, for the configuration's profile."""
    profile = project.resolve_profile(configuration)
    if profile is None:
        raise McdaError("the project has no weight profile")
    check = elicitation_check(project, profile.id)
    previews = [
        ProposalPreview(
            mode="current",
            label="Stated uncertainty now",
            robustness=robustness.screening_robustness(project, configuration),
        )
    ]
    for proposal in check.proposals:
        trial = project.model_copy(update={"smaa": proposal.smaa}, deep=True)
        previews.append(
            ProposalPreview(
                mode=proposal.mode,
                label=proposal.label,
                robustness=robustness.screening_robustness(trial, configuration),
            )
        )
    return ElicitationPreview(
        project_fingerprint=project.fingerprint(), profile_id=profile.id, previews=previews
    )
