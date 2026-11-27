"""Trade-off elicitation from a project: questions, the next one to ask, and the implied weights.

Read-only: the project is never changed, and answers live in a ``tradeoff-answers`` sidecar
outside it. The method is ``methods/tradeoff.py``; this service supplies the prior from a weight
profile, the questions from the criteria's own levels, and the report.

Questions (PAPRIKA style): two alternatives that differ on exactly two criteria, one better on the
first and worse on the second, at the levels of each criterion's scale: the labels of a labelled
scale, the points of a piecewise-linear value function, and the ends and middle of a linear or
direct-rating one. Optionally, pairs of the project's own alternatives.

Design: ``docs/design/tradeoff-elicitation.md``.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mcda_studio.domain import tradeoff_answers as ta
from mcda_studio.domain import weight_samples as ws
from mcda_studio.domain.common import DomainModel
from mcda_studio.domain.criteria import (
    DirectRatingValueFunction,
    LinearValueFunction,
    PiecewiseLinearValueFunction,
    QualitativeValueFunction,
)
from mcda_studio.domain.errors import DomainError, McdaError, UnknownReferenceError
from mcda_studio.domain.preferences import AnalysisConfiguration, WeightProfile
from mcda_studio.domain.project import Project
from mcda_studio.methods import tradeoff as tm
from mcda_studio.methods.mavt.aggregation import PreparedMavt, prepare_mavt
from mcda_studio.methods.mavt.value_functions import build_evaluator
from mcda_studio.methods.nested_dirichlet import prior_for_profile
from mcda_studio.methods.sorting.classes import class_index, lower_boundaries
from mcda_studio.services.elicitation_service import CONVICTION_HALF_WIDTH

PriorChoice = Literal["auto", "ranking", "unknown", "stated", "low", "medium", "high"]
Mode = Literal["auto", "exact", "noisy"]

SHEET_QUESTIONS = 20
SHEET_COLUMNS = [
    "id", "criterion_1_id", "criterion_1", "a_level_1", "b_level_1", "criterion_2_id",
    "criterion_2", "a_level_2", "b_level_2", "choice", "note",
]  # fmt: skip
_FORMULA = ("=", "+", "-", "@", "\t", "\r")

LABEL = (
    "Implied weights describe what is consistent with these answers under the stated prior and "
    "an additive value model; they are a check on the stated weights, not a replacement for them."
)
LIMITATIONS = [
    "Exact mode assumes every answer is the person's considered preference: one slip removes the "
    "truth from the region and nothing in exact mode can bring it back. Noisy mode allows slips.",
    "Answers inform weights only along the contrasts asked; elsewhere the prior decides.",
    "Each question changes two criteria with everything else equal, so an answer reads the "
    "additive model's trade-off at those levels; it says nothing about interactions the model "
    "leaves out.",
    "Particles after rejuvenation are correlated, so intervals from exact mode carry more "
    "Monte Carlo error than the particle count suggests.",
]
TRADEOFF_LABEL = ws.TRADEOFF_LABEL


# ---- the prior from a weight profile ------------------------------------------------------------


class PriorGroupInfo(DomainModel):
    parent_id: str | None
    name: str
    kind: str
    children: list[str]
    held: list[str]
    detail: str


class PriorInfo(DomainModel):
    choice: str
    kind: str  # ranking, unknown, stated or conviction
    conviction: str | None
    source: str
    groups: list[PriorGroupInfo]


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


def _held_map(project: Project) -> dict[str | None, list[str]]:
    held: dict[str | None, list[str]] = {}
    if project.smaa is not None:
        for group in project.smaa.groups:
            held[group.parent_id] = list(group.held)
    return held


def _structure(
    project: Project, local: dict[str, float], held_map: dict[str | None, list[str]]
) -> list[tuple[str | None, list[str], list[str], float]]:
    """Varying groups as ``make_prior`` finds them: (parent, free, held, share)."""
    tree = project.tree()
    out = []
    for parent in [None, *[n for n in tree.all_ids() if not tree.is_leaf(n)]]:
        kids = tree.child_ids(parent)
        if not kids:
            continue
        share_of_parent = 1.0 if parent is None else math.prod(local[n] for n in tree.path(parent))
        if share_of_parent == 0.0:
            continue
        listed = set(held_map.get(parent, []))
        held = [k for k in kids if local[k] == 0.0 or k in listed]
        free = [k for k in kids if k not in held]
        if len(free) < 2:
            continue
        out.append((parent, free, held, math.fsum(local[k] for k in free)))
    return out


def _swing_order(free: list[str], local: dict[str, float]) -> list[str]:
    """Children best first by local weight; ties keep the tree's order."""
    position = {k: i for i, k in enumerate(free)}
    return sorted(free, key=lambda k: (-local[k], position[k]))


def build_prior(
    project: Project, profile: WeightProfile, choice: PriorChoice = "auto"
) -> tuple[tm.ConstrainedPrior, PriorInfo]:
    """The prior over global weights for a profile.

    ``auto``: the profile's elicitation record decides: its conviction if it has one, otherwise the
    ranking prior with the record's order where it gives one and the swing ratings' order
    elsewhere. ``ranking``: that ranking prior. ``unknown``: every varying group uniform.
    ``stated``: the project's stated SMAA-2 weight groups. ``low``/``medium``/``high``: the
    conviction prior at that level."""
    from mcda_studio.methods.mavt.weighting import compute_weights

    tree = project.tree()
    local = compute_weights(tree, profile.judgements).local
    names = {n.id: n.name for n in project.criteria}
    held_map = _held_map(project)
    record = profile.elicitation
    if choice == "auto":
        choice = record.conviction if record is not None and record.conviction else "ranking"
        source = (
            "the profile's elicitation record"
            if choice != "ranking"
            else ("no conviction recorded: ranking prior")
        )
    else:
        source = "stated with the request"
    groups: list[tm.LatentGroup] = []
    info: list[PriorGroupInfo] = []

    def where(parent: str | None) -> str:
        return "the top level" if parent is None else names[parent]

    if choice in ("low", "medium", "high"):
        nested = prior_for_profile(
            tree, local, profile.judgements, held_map, half_width=CONVICTION_HALF_WIDTH[choice]
        )
        for g in nested.groups:
            alpha = tuple(g.kappa * c for c in g.centre)
            groups.append(
                tm.LatentGroup(g.parent_id, "dirichlet", g.children, g.share, g.held, alpha)
            )
            info.append(
                PriorGroupInfo(
                    parent_id=g.parent_id,
                    name=where(g.parent_id),
                    kind="dirichlet",
                    children=list(g.children),
                    held=list(g.held),
                    detail=f"Dirichlet(kappa * stated local weights), kappa = {g.kappa:g} "
                    f"({g.kappa_method})",
                )
            )
        kind = "conviction"
    elif choice in ("ranking", "unknown"):
        orders = {g.parent_id: list(g.order) for g in (record.groups if record else [])}
        for parent, free, held, share in _structure(project, local, held_map):
            if choice == "unknown":
                groups.append(
                    tm.LatentGroup(parent, "dirichlet", tuple(free), share, tuple(held),
                                   tuple(1.0 for _ in free))
                )  # fmt: skip
                detail = "uniform over every split (Dirichlet(1, ..., 1))"
                info.append(
                    PriorGroupInfo(parent_id=parent, name=where(parent), kind="dirichlet",
                                   children=free, held=held, detail=detail)
                )  # fmt: skip
                continue
            stated = [k for k in orders.get(parent, []) if k in free]
            if len(stated) == len(free):
                order, from_what = stated, "the elicitation record's order"
            else:
                order, from_what = _swing_order(free, local), "the swing ratings' order"
            groups.append(
                tm.LatentGroup(parent, "ordered", tuple(order), share, tuple(held),
                               tuple(1.0 for _ in order))
            )  # fmt: skip
            info.append(
                PriorGroupInfo(
                    parent_id=parent,
                    name=where(parent),
                    kind="ordered",
                    children=order,
                    held=held,
                    detail="uniform over every split that keeps the order "
                    + " > ".join(names[k] for k in order)
                    + f" ({from_what})",
                )
            )
        kind = choice
    else:  # stated
        if project.smaa is None or not project.smaa.groups:
            raise McdaError("the project states no SMAA-2 weight groups to use as the prior")
        by_parent = {g.parent_id: g for g in project.smaa.groups}
        for parent in [None, *[n for n in tree.all_ids() if not tree.is_leaf(n)]]:
            spec = by_parent.get(parent)
            if spec is None:
                continue
            share_of_parent = (
                1.0 if parent is None else math.prod(local[n] for n in tree.path(parent))
            )
            if share_of_parent == 0.0:
                continue
            kids = tree.child_ids(parent)
            held = [k for k in kids if k in spec.held]
            varying = [k for k in kids if k not in held]
            if spec.kind == "rating_intervals":
                zero = [k for k in varying if tuple(spec.intervals.get(k, (1.0, 1.0))) == (0, 0)]
                for k in zero:
                    if local[k] != 0.0:
                        raise McdaError(
                            f"{names[k]!r} has the interval [0, 0] but a nonzero rating; the "
                            "stated prior cannot hold it"
                        )
                held = held + zero
                varying = [k for k in varying if k not in zero]
            share = 1.0 - math.fsum(local[k] for k in held)
            if len(varying) < 2:
                continue
            if spec.kind == "unknown":
                groups.append(
                    tm.LatentGroup(parent, "dirichlet", tuple(varying), share, tuple(held),
                                   tuple(1.0 for _ in varying))
                )  # fmt: skip
                detail = "uniform over every split (stated 'unknown')"
                children = varying
            elif spec.kind == "ranking":
                order = [k for k in spec.order if k in varying]
                groups.append(
                    tm.LatentGroup(parent, "ordered", tuple(order), share, tuple(held),
                                   tuple(1.0 for _ in order))
                )  # fmt: skip
                detail = "uniform over every split that keeps the stated order"
                children = order
            else:
                lo = tuple(float(spec.intervals[k][0]) for k in varying)
                hi = tuple(float(spec.intervals[k][1]) for k in varying)
                if any(a == b for a, b in zip(lo, hi, strict=True)):
                    raise McdaError("a stated rating interval of zero width cannot be a prior")
                groups.append(
                    tm.LatentGroup(parent, "intervals", tuple(varying), share, tuple(held),
                                   lower=lo, upper=hi)
                )  # fmt: skip
                detail = "each swing rating uniform within its stated interval, then normalised"
                children = varying
            info.append(
                PriorGroupInfo(parent_id=parent, name=where(parent), kind=groups[-1].kind,
                               children=children, held=held, detail=detail)
            )  # fmt: skip
        kind = "stated"
    if not groups:
        raise McdaError("no sibling group of this profile has two or more weights to vary")
    prior = tm.ConstrainedPrior(
        tree.all_ids(),
        tree.leaves(),
        {leaf: tree.path(leaf) for leaf in tree.leaves()},
        local,
        groups,
    )
    conviction = choice if choice in ("low", "medium", "high") else None
    return prior, PriorInfo(
        choice=choice, kind=kind, conviction=conviction, source=source, groups=info
    )


# ---- levels and questions -----------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    raw: str | float
    shown: str
    value: float


@dataclass(frozen=True)
class Candidate:
    a: dict[str, str | float]
    b: dict[str, str | float]
    alternatives: tuple[str, str] | None
    contrast: np.ndarray  # v(A) - v(B), leaf order

    def key(self) -> tuple[object, ...]:
        return question_key(self.a, self.b, self.alternatives)


def _norm(raw: str | float) -> str:
    return repr(float(raw)) if isinstance(raw, int | float) else str(raw)


def question_key(
    a: dict[str, str | float], b: dict[str, str | float], alternatives: Sequence[str] | None
) -> tuple[object, ...]:
    """The same question whichever side is called A."""
    if alternatives is not None:
        return ("alternatives", tuple(sorted(alternatives)))
    one = tuple(sorted((k, _norm(a[k]), _norm(b[k])) for k in a))
    two = tuple(sorted((k, _norm(b[k]), _norm(a[k])) for k in a))
    return ("levels", min(one, two))


@dataclass
class Context:
    """What every call needs from one project and profile."""

    project: Project
    profile: WeightProfile
    prep: PreparedMavt
    prior: tm.ConstrainedPrior
    prior_info: PriorInfo
    levels: dict[str, list[Level]]
    names: dict[str, str]
    alt_names: dict[str, str]

    @property
    def leaf_ids(self) -> list[str]:
        return list(self.prior.leaf_ids)

    @property
    def partial(self) -> np.ndarray:
        # prepare_mavt's leaves are the tree's leaves, the prior's order.
        return self.prep.partial

    def value(self, leaf: str, raw: str | float) -> float:
        """The partial value of one level, through the project's value function."""
        node = self.prep.tree.node(leaf)
        vf = node.value_function
        if vf is None:
            raise McdaError(f"criterion {leaf!r} has no value function")
        evaluator = build_evaluator(vf, self.prep.domains[leaf])
        if isinstance(vf, QualitativeValueFunction):
            if not isinstance(raw, str) or raw not in vf.mapping:
                raise McdaError(f"{raw!r} is not a level of {self.names[leaf]!r}")
            return evaluator(raw)[0]
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise McdaError(f"{raw!r} is not a number for {self.names[leaf]!r}") from None
        try:
            return evaluator(number)[0]
        except DomainError as exc:
            raise McdaError(f"{self.names[leaf]!r}: {exc}") from None

    def contrast(self, answer: ta.TradeoffAnswer) -> np.ndarray:
        """``v(chosen) - v(other)``; zero for a skipped answer."""
        index = {leaf: j for j, leaf in enumerate(self.leaf_ids)}
        d = np.zeros(len(self.leaf_ids))
        if answer.alternatives is not None:
            alts = self.prep.alt_ids
            for alt in answer.alternatives:
                if alt not in alts:
                    raise McdaError(f"answer {answer.id}: unknown alternative {alt!r}")
            d = (
                self.partial[alts.index(answer.alternatives[0])]
                - self.partial[alts.index(answer.alternatives[1])]
            )
        else:
            for leaf in answer.a:
                if leaf not in index:
                    raise McdaError(f"answer {answer.id}: {leaf!r} is not a criterion leaf")
                d[index[leaf]] = self.value(leaf, answer.a[leaf]) - self.value(leaf, answer.b[leaf])
        if answer.choice == "skip":
            return np.zeros(len(self.leaf_ids))
        return d if answer.choice == "A" else -d


def _leaf_levels(project: Project, prep: PreparedMavt, leaf: str) -> list[Level]:
    node = prep.tree.node(leaf)
    vf = node.value_function
    assert vf is not None and node.scale is not None
    evaluator = build_evaluator(vf, prep.domains[leaf])
    raws: list[str | float] = []
    if isinstance(vf, QualitativeValueFunction):
        raws = list(vf.mapping)
    elif isinstance(vf, PiecewiseLinearValueFunction):
        raws = [float(p.x) for p in vf.points]
    elif isinstance(vf, LinearValueFunction | DirectRatingValueFunction):
        if isinstance(vf, DirectRatingValueFunction):
            lo, hi = float(vf.min_rating), float(vf.max_rating)
        else:
            domain = prep.domains[leaf]
            if domain is None:
                return []
            lo, hi = float(domain.lower), float(domain.upper)
        raws = [lo, (lo + hi) / 2.0, hi]
    unit = node.scale.unit if getattr(node.scale, "unit", None) else ""
    levels: list[Level] = []
    seen: set[float] = set()
    for raw in raws:
        value = evaluator(raw)[0]
        if value in seen:
            continue  # a second label at the same value adds no contrast
        seen.add(value)
        shown = raw if isinstance(raw, str) else f"{raw:g}" + (f" {unit}" if unit else "")
        levels.append(Level(raw=raw, shown=shown, value=value))
    return sorted(levels, key=lambda lv: lv.value)


def context(project: Project, profile_id: str | None, prior: PriorChoice = "auto") -> Context:
    profile = _profile(project, profile_id)
    cfg = AnalysisConfiguration(method="mavt", weight_profile_id=profile.id)
    prep = prepare_mavt(project, cfg)
    if prep.leaf_ids != project.tree().leaves():
        raise McdaError("internal: leaf order differs")  # pragma: no cover
    constrained, info = build_prior(project, profile, prior)
    names = {n.id: n.name for n in project.criteria}
    return Context(
        project=project,
        profile=profile,
        prep=prep,
        prior=constrained,
        prior_info=info,
        levels={leaf: _leaf_levels(project, prep, leaf) for leaf in prep.leaf_ids},
        names=names,
        alt_names={a.id: a.name for a in project.alternatives},
    )


def candidates(ctx: Context, real_pairs: bool = False) -> list[Candidate]:
    """Every two-criterion question on the varying leaves, then (optionally) real pairs.

    For leaves ``i < j`` (tree order) and levels ``lo_i < hi_i``, ``lo_j < hi_j``: A has
    ``(hi_i, lo_j)`` and B ``(lo_i, hi_j)``. Questions with the same value contrast keep the
    first."""
    leaves = [leaf for j, leaf in enumerate(ctx.leaf_ids) if ctx.prior.varying[j]]
    index = {leaf: j for j, leaf in enumerate(ctx.leaf_ids)}
    out: list[Candidate] = []
    seen: set[tuple[int, int, float, float]] = set()
    for x, first in enumerate(leaves):
        for second in leaves[x + 1 :]:
            for lo1, hi1 in _level_pairs(ctx.levels[first]):
                for lo2, hi2 in _level_pairs(ctx.levels[second]):
                    d1, d2 = hi1.value - lo1.value, hi2.value - lo2.value
                    key = (index[first], index[second], round(d1, 12), round(d2, 12))
                    if key in seen:
                        continue
                    seen.add(key)
                    contrast = np.zeros(len(ctx.leaf_ids))
                    contrast[index[first]] = d1
                    contrast[index[second]] = -d2
                    out.append(
                        Candidate(
                            a={first: hi1.raw, second: lo2.raw},
                            b={first: lo1.raw, second: hi2.raw},
                            alternatives=None,
                            contrast=contrast,
                        )
                    )
    if real_pairs:
        alts = ctx.prep.alt_ids
        for x, first in enumerate(alts):
            for y in range(x + 1, len(alts)):
                contrast = ctx.partial[x] - ctx.partial[y]
                if np.any(contrast != 0.0):
                    out.append(Candidate({}, {}, (first, alts[y]), contrast))
    return out


def _level_pairs(levels: list[Level]) -> list[tuple[Level, Level]]:
    return [(lo, hi) for i, lo in enumerate(levels) for hi in levels[i + 1 :]]


def describe(ctx: Context, a: dict[str, str | float], b: dict[str, str | float],
             alternatives: Sequence[str] | None) -> str:  # fmt: skip
    """The question in words."""
    if alternatives is not None:
        return (
            f"A: {ctx.alt_names[alternatives[0]]}. B: {ctx.alt_names[alternatives[1]]}. "
            "Which would you rather back, on the criteria as rated?"
        )

    def side(levels: dict[str, str | float]) -> str:
        return "; ".join(f"{ctx.names[k]}: {_shown(ctx, k, levels[k])}" for k in levels)

    return f"A: {side(a)}. B: {side(b)}. Everything else is the same. Which do you prefer?"


def _shown(ctx: Context, leaf: str, raw: str | float) -> str:
    for level in ctx.levels.get(leaf, []):
        if _norm(level.raw) == _norm(raw):
            return level.shown
    return str(raw)


# ---- answers ------------------------------------------------------------------------------------


def check_answers(ctx: Context, answers: ta.TradeoffAnswers) -> list[str]:
    """Warnings for answers given against another version of the project; refusals raise."""
    warnings: list[str] = []
    if answers.project_id != ctx.project.id:
        raise McdaError(
            f"the answers are for project {answers.project_id!r}, not {ctx.project.id!r}"
        )
    if answers.weight_profile_id != ctx.profile.id:
        raise McdaError(
            f"the answers were given against profile {answers.weight_profile_id!r}, not "
            f"{ctx.profile.id!r}"
        )
    if answers.project_fingerprint != ctx.project.fingerprint():
        warnings.append(
            "the project has changed since these answers were given; the levels are read "
            "through today's value functions"
        )
    for answer in answers.answers:
        ctx.contrast(answer)  # refuses unknown levels and criteria
    return warnings


def answers_digest(answers: ta.TradeoffAnswers) -> str:
    text = json.dumps(
        answers.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def answers_from_sheet(
    ctx: Context, text: str, *, answered_by: str, answered_on: str | None
) -> tuple[ta.TradeoffAnswers, int]:
    """Answers from a filled sheet; a blank choice counts as skip. Returns the unanswered count."""
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in SHEET_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise McdaError("the sheet lacks the columns " + ", ".join(missing))
    out: list[ta.TradeoffAnswer] = []
    blank = 0
    problems: list[str] = []
    for line, row in enumerate(reader, start=2):
        row = {k: _unsafe(v or "").strip() for k, v in row.items() if k is not None}
        choice = row["choice"].strip()
        normal = {"a": "A", "b": "B", "skip": "skip", "": "skip"}.get(choice.lower())
        if normal is None:
            problems.append(f"line {line}: choice must be A, B or skip, not {choice!r}")
            continue
        blank += choice == ""
        a: dict[str, str | float] = {}
        b: dict[str, str | float] = {}
        for n in ("1", "2"):
            leaf = row[f"criterion_{n}_id"]
            a[leaf] = _level_from_text(ctx, leaf, row[f"a_level_{n}"])
            b[leaf] = _level_from_text(ctx, leaf, row[f"b_level_{n}"])
        try:
            out.append(
                ta.TradeoffAnswer(id=row["id"], a=a, b=b, choice=normal, note=row["note"])  # type: ignore[arg-type]
            )
        except ValueError as exc:
            problems.append(f"line {line}: {exc}")
    if problems:
        raise McdaError("; ".join(problems))
    answers = ta.TradeoffAnswers(
        kind="tradeoff-answers",
        version=1,
        project_id=ctx.project.id,
        project_fingerprint=ctx.project.fingerprint(),
        weight_profile_id=ctx.profile.id,
        answered_by=answered_by,
        answered_on=answered_on,
        source="person",
        answers=out,
    )
    return answers, blank


def _unsafe(value: str) -> str:
    return value[1:] if value.startswith("'") and value[1:].startswith(_FORMULA) else value


def _level_from_text(ctx: Context, leaf: str, text: str) -> str | float:
    node = ctx.prep.tree.node(leaf) if leaf in ctx.prep.tree else None
    if node is None:
        raise McdaError(f"{leaf!r} is not a criterion")
    if isinstance(node.value_function, QualitativeValueFunction):
        return text
    for level in ctx.levels.get(leaf, []):
        if level.shown == text:
            return level.raw
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        raise McdaError(f"{text!r} is not a number for {ctx.names[leaf]!r}") from None


# ---- next question ------------------------------------------------------------------------------


class QuestionOut(DomainModel):
    id: str
    a: dict[str, str | float]
    b: dict[str, str | float]
    alternatives: list[str] | None = None
    text: str
    p_a_over_b: float


class ConflictOut(DomainModel):
    answer_id: str
    contradicts: list[str]
    note: str


class NextOut(DomainModel):
    status: Literal["ask", "stop"]
    reason: str
    asked: int
    question: QuestionOut | None
    p_first: dict[str, float]
    conflicts: list[ConflictOut]
    warnings: list[str]


@dataclass
class _Run:
    state: tm.SmcState
    history: list[np.ndarray]
    kept: list[int]
    conflicts: list[ConflictOut]
    answered: int
    asked_keys: set[tuple[object, ...]]


def _run_answers(
    ctx: Context, answers: Sequence[ta.TradeoffAnswer], settings: tm.SmcSettings
) -> _Run:
    state = tm.SmcState.start(ctx.prior, settings)
    history = [tm.first_probabilities(state.weights, ctx.partial)]
    kept: list[int] = []
    conflicts: list[ConflictOut] = []
    keys: set[tuple[object, ...]] = set()
    answered = 0
    contrasts: dict[int, np.ndarray] = {}
    for k, answer in enumerate(answers):
        keys.add(question_key(answer.a, answer.b, answer.alternatives))
        if answer.choice == "skip":
            continue
        answered += 1
        contrast = ctx.contrast(answer)
        contrasts[k] = contrast
        if state.add(contrast, settings.moves):
            kept.append(k)
            history.append(tm.first_probabilities(state.weights, ctx.partial))
        else:
            conflicts.append(_conflict(ctx, answers, k, kept, contrasts))
    return _Run(state, history, kept, conflicts, answered, keys)


def _conflict(
    ctx: Context,
    answers: Sequence[ta.TradeoffAnswer],
    k: int,
    kept: list[int],
    contrasts: dict[int, np.ndarray],
) -> ConflictOut:
    """The latest answer and the earlier answers it contradicts one to one (exact, on the
    simplex of the leaves the prior moves). No LP solver is a runtime dependency, so a minimal
    set of three or more answers is not searched for."""
    support = ctx.prior.varying
    against = [
        answers[j].id for j in kept if tm.pair_infeasible(contrasts[k], contrasts[j], support)
    ]
    if against:
        note = (
            "no weights satisfy this answer together with each listed one; re-ask these questions"
        )
    else:
        note = (
            "no particle satisfies this answer with the earlier ones, and no single earlier "
            "answer contradicts it; the answers may conflict jointly, or leave a region too "
            "small for the particles. Re-ask it, or fit in noisy mode"
        )
    return ConflictOut(answer_id=answers[k].id, contradicts=against, note=note)


def next_question(
    ctx: Context,
    answers: ta.TradeoffAnswers | None,
    settings: tm.SmcSettings,
    *,
    real_pairs: bool = False,
    mode: Literal["adaptive", "random"] = "adaptive",
) -> NextOut:
    warnings = check_answers(ctx, answers) if answers is not None else []
    given = answers.answers if answers is not None else []
    run = _run_answers(ctx, given, settings)
    p_first = {a: float(p) for a, p in zip(ctx.prep.alt_ids, run.history[-1], strict=True)}
    if tm.should_stop(run.history, settings.particles, run.answered):
        reason = (
            f"{tm.MAX_QUESTIONS} questions asked"
            if run.answered >= tm.MAX_QUESTIONS
            else f"P(first) moved less than its 99% sampling noise over the last "
            f"{tm.STOP_WINDOW} answers"
        )
        return NextOut(status="stop", reason=reason, asked=run.answered, question=None,
                       p_first=p_first, conflicts=run.conflicts, warnings=warnings)  # fmt: skip
    pool = candidates(ctx, real_pairs)
    probs = tm.answer_probabilities(run.state.weights, np.array([c.contrast for c in pool]))
    asked = {i for i, c in enumerate(pool) if c.key() in run.asked_keys}
    rng = np.random.Generator(np.random.PCG64(settings.seed).jumped(1))
    pick = tm.select_question(probs, asked, mode=mode, rng=rng)
    if pick is None:
        return NextOut(
            status="stop",
            reason="every remaining question's answer is already implied by the answers so far",
            asked=run.answered, question=None, p_first=p_first, conflicts=run.conflicts,
            warnings=warnings,
        )  # fmt: skip
    chosen = pool[pick]
    used = {a.id for a in given}
    number = len(given) + 1
    while f"q{number}" in used:
        number += 1
    question = QuestionOut(
        id=f"q{number}",
        a=chosen.a,
        b=chosen.b,
        alternatives=list(chosen.alternatives) if chosen.alternatives else None,
        text=describe(ctx, chosen.a, chosen.b, chosen.alternatives),
        p_a_over_b=float(probs[pick]),
    )
    return NextOut(status="ask", reason="closest to an even split under the current weights"
                   if mode == "adaptive" else "chosen at random among the open questions",
                   asked=run.answered, question=question, p_first=p_first,
                   conflicts=run.conflicts, warnings=warnings)  # fmt: skip


# ---- fit ----------------------------------------------------------------------------------------


class LeafRow(DomainModel):
    criterion_id: str
    name: str
    varies: bool
    stated: float
    prior_median: float
    prior_low: float
    prior_high: float
    implied_median: float
    implied_low: float  # 5%
    implied_high: float  # 95%
    gap: float  # implied median minus stated


class ConfidenceRow(DomainModel):
    alternative_id: str
    name: str
    p_first_before: float
    p_first_after: float
    classes_before: dict[str, float]
    classes_after: dict[str, float]


class NoisyQuantity(DomainModel):
    name: str
    rhat: float | None
    ess_bulk: float | None


class NoisyDiagnostics(DomainModel):
    """The calibration sampler's gate for a noisy fit: R-hat below 1.01, bulk ESS at least 400."""

    converged: bool
    quantities: list[NoisyQuantity]
    acceptance: dict[str, list[float]]


class TradeoffReport(DomainModel):
    status: Literal["exact", "noisy", "not_converged", "conflict"]
    mode_asked: str
    project_id: str
    project_fingerprint: str
    weight_profile_id: str
    answers_sha256: str
    answered_by: str
    source: str
    prior: PriorInfo
    answers: int
    skipped: int
    kept: int
    conflicts: list[ConflictOut]
    seed: int
    particles: int
    leaves: list[LeafRow]
    confidence: list[ConfidenceRow]
    smc_acceptance: list[float]
    diagnostics: NoisyDiagnostics | None = None
    beta: list[float] | None = None  # noisy mode: median and 90% interval
    warnings: list[str]
    label: str = LABEL
    limitations: list[str] = LIMITATIONS


@dataclass
class TradeoffFit:
    report: TradeoffReport
    draws: np.ndarray | None  # (n, leaves); None when not converged or in conflict
    algorithm: str
    noisy: tm.NoisyFit | None


def _quantiles(x: np.ndarray) -> tuple[float, float, float]:
    q = np.quantile(x, [0.5, 0.05, 0.95])
    return float(q[0]), float(q[1]), float(q[2])


def _classes(ctx: Context, weights: np.ndarray) -> list[dict[str, float]]:
    sorting = ctx.project.sorting
    n_alt = len(ctx.prep.alt_ids)
    if sorting is None:
        return [{} for _ in range(n_alt)]
    bounds = lower_boundaries(sorting)
    values = weights @ ctx.partial.T
    out = []
    for a in range(n_alt):
        idx = np.array([class_index(float(v), bounds) for v in values[:, a]])
        counts = np.bincount(idx, minlength=len(sorting.classes))
        out.append(
            {c.id: float(counts[i] / weights.shape[0]) for i, c in enumerate(sorting.classes)}
        )
    return out


def fit(
    ctx: Context,
    answers: ta.TradeoffAnswers,
    *,
    mode: Mode = "auto",
    settings: tm.SmcSettings | None = None,
    noisy: tm.NoisyFitter | None = None,
) -> TradeoffFit:
    """Implied weights from the answers.

    ``exact``: the restricted prior; refuses (status ``conflict``) when an answer leaves no
    particle. ``noisy``: Bradley-Terry through the calibration sampler, passed in as ``noisy``
    (``calibration.tradeoff.fit_noisy``). ``auto``: exact, falling back to noisy when there is a
    conflict."""
    settings = settings or tm.SmcSettings()
    warnings = check_answers(ctx, answers)
    run = _run_answers(ctx, answers.answers, settings)
    before = tm.SmcState.start(ctx.prior, settings).weights
    skipped = sum(1 for a in answers.answers if a.choice == "skip")
    use_noisy = mode == "noisy" or (mode == "auto" and bool(run.conflicts))
    diagnostics: NoisyDiagnostics | None = None
    beta: list[float] | None = None
    draws: np.ndarray
    algorithm = tm.ALGORITHM
    noisy_fit: tm.NoisyFit | None = None
    status: Literal["exact", "noisy", "not_converged", "conflict"]
    if use_noisy:
        if noisy is None:
            raise McdaError("noisy mode runs only from the command line (mcda elicit tradeoff fit)")
        contrasts = np.array([ctx.contrast(a) for a in answers.answers if a.choice != "skip"])
        if contrasts.size == 0:
            raise McdaError("noisy mode needs at least one answer that is not skipped")
        try:
            nested = tm.ordered_nested_prior(ctx.prior)
        except ValueError as exc:
            raise McdaError(str(exc)) from None
        noisy_fit = noisy(nested, contrasts, settings.seed)
        diagnostics = NoisyDiagnostics(
            converged=noisy_fit.converged,
            quantities=[
                NoisyQuantity(name=n, rhat=r, ess_bulk=e) for n, r, e in noisy_fit.quantities
            ],
            acceptance=noisy_fit.acceptance,
        )
        draws = noisy_fit.draws
        b = noisy_fit.beta
        beta = [float(np.median(b)), float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))]
        status = "noisy" if noisy_fit.converged else "not_converged"
        algorithm = noisy_fit.algorithm
    elif run.conflicts:
        draws = run.state.weights
        status = "conflict"
    else:
        draws = run.state.weights
        status = "exact"
    stated = ctx.prep.global_
    rows = []
    for j, leaf in enumerate(ctx.leaf_ids):
        pm, pl, ph = _quantiles(before[:, j])
        im, il, ih = _quantiles(draws[:, j])
        rows.append(
            LeafRow(
                criterion_id=leaf, name=ctx.names[leaf], varies=bool(ctx.prior.varying[j]),
                stated=stated[leaf], prior_median=pm, prior_low=pl, prior_high=ph,
                implied_median=im, implied_low=il, implied_high=ih, gap=im - stated[leaf],
            )
        )  # fmt: skip
    p_before = tm.first_probabilities(before, ctx.partial)
    p_after = tm.first_probabilities(draws, ctx.partial)
    c_before = _classes(ctx, before)
    c_after = _classes(ctx, draws)
    confidence = [
        ConfidenceRow(
            alternative_id=a, name=ctx.alt_names[a], p_first_before=float(p_before[i]),
            p_first_after=float(p_after[i]), classes_before=c_before[i], classes_after=c_after[i],
        )
        for i, a in enumerate(ctx.prep.alt_ids)
    ]  # fmt: skip
    result = TradeoffReport(
        status=status,
        mode_asked=mode,
        project_id=ctx.project.id,
        project_fingerprint=ctx.project.fingerprint(),
        weight_profile_id=ctx.profile.id,
        answers_sha256=answers_digest(answers),
        answered_by=answers.answered_by,
        source=answers.source,
        prior=ctx.prior_info,
        answers=len(answers.answers),
        skipped=skipped,
        kept=len(run.kept),
        conflicts=run.conflicts,
        seed=settings.seed,
        particles=settings.particles,
        leaves=rows,
        confidence=confidence,
        smc_acceptance=[round(a, 6) for a in run.state.acceptance],
        diagnostics=diagnostics,
        beta=beta,
        warnings=warnings,
    )
    usable = status in ("exact", "noisy")
    return TradeoffFit(
        report=result,
        draws=draws if usable else None,
        algorithm=algorithm,
        noisy=noisy_fit,
    )


def weight_samples(
    ctx: Context, answers: ta.TradeoffAnswers, result: TradeoffFit
) -> ws.WeightSamples:
    """The implied weights as a ``weight-samples`` sidecar with ``basis: "tradeoff"``."""
    if result.draws is None:
        raise McdaError("no samples: the fit is in conflict or did not converge")
    report = result.report
    draws = result.draws
    if report.source != "person":
        raise McdaError("simulated answers never become weight samples")
    prior = ctx.prior
    groups = []
    for g, info in zip(prior.groups, report.prior.groups, strict=True):
        if g.kind == "intervals":
            mid = [(lo + hi) / 2.0 for lo, hi in zip(g.lower, g.upper, strict=True)]
            total = math.fsum(mid)
            centre, kappa = [m / total for m in mid], 0.0
        else:
            kappa = math.fsum(g.alpha)
            centre = [a / kappa for a in g.alpha]
        groups.append(
            ws.GroupPrior(
                parent_id=g.parent_id, children=list(g.children), centre=centre, kappa=kappa,
                kappa_method=info.detail, share=g.share, held=list(g.held),
            )
        )  # fmt: skip
    varying = {leaf for leaf, v in zip(prior.leaf_ids, prior.varying, strict=True) if v}
    fixed = [
        ws.FixedGroupPrior(parent_id=p, children=kids, reason="not varied by this prior")
        for p, kids in _fixed_groups(ctx, varying)
    ]
    centre_global = {leaf: ctx.prep.global_[leaf] for leaf in prior.leaf_ids}
    sampler_spec: ws.SamplerSpec
    diag: ws.DiagnosticsSpec
    if result.noisy is None:
        sampler_spec = ws.SamplerSpec(
            algorithm=result.algorithm, chains=1, warmup=0, draws=draws.shape[0],
            samples=draws.shape[0], numpy_version=np.__version__,
            python_version=platform.python_version(),
        )  # fmt: skip
        diag = ws.DiagnosticsSpec(
            converged=True, rhat_max=1.01, ess_min=400.0, quantities=[],
            acceptance={"rejuvenation": list(report.smc_acceptance)},
        )  # fmt: skip
    else:
        nf = result.noisy
        sampler_spec = ws.SamplerSpec(
            algorithm=result.algorithm, chains=nf.chains, warmup=nf.warmup,
            draws=nf.draws_per_chain, samples=nf.samples, numpy_version=np.__version__,
            python_version=platform.python_version(),
        )  # fmt: skip
        diag = ws.DiagnosticsSpec(
            converged=nf.converged, rhat_max=1.01, ess_min=400.0,
            quantities=[
                ws.QuantityDiagnosticsSpec(name=n, rhat=r, ess_bulk=e) for n, r, e in nf.quantities
            ],
            acceptance=nf.acceptance,
        )  # fmt: skip
    rows = []
    for row in draws:
        clean = [max(0.0, float(x)) for x in row]
        total = math.fsum(clean)
        rows.append([x / total for x in clean])
    return ws.WeightSamples(
        kind="weight-samples",
        version=1,
        project_id=ctx.project.id,
        project_fingerprint=ctx.project.fingerprint(),
        weight_profile_id=ctx.profile.id,
        basis="tradeoff",
        diagnostic=False,
        usable=True,
        review_states=[],
        prior=ws.PriorSpec(
            conviction=report.prior.conviction,  # type: ignore[arg-type]
            conviction_source=(
                f"{report.prior.kind} prior ({report.prior.source}); {report.kept} trade-off "
                f"answers by {answers.answered_by}, tradeoff-answers sha256 "
                f"{report.answers_sha256}; {report.status} mode"
            ),
            half_width=CONVICTION_HALF_WIDTH[report.prior.conviction]
            if report.prior.conviction
            else None,
            groups=groups,
            fixed_groups=fixed,
            centre_global=centre_global,
        ),
        seed=report.seed,
        sampler=sampler_spec,
        diagnostics=diag,
        leaf_ids=list(prior.leaf_ids),
        samples=rows,
        label=ws.TRADEOFF_LABEL,
        limitations=list(ws.TRADEOFF_LIMITATIONS),
    )


def _fixed_groups(ctx: Context, varying: set[str]) -> list[tuple[str | None, list[str]]]:
    tree = ctx.prep.tree
    moved = {g.parent_id for g in ctx.prior.groups}
    out = []
    for parent in [None, *[n for n in tree.all_ids() if not tree.is_leaf(n)]]:
        kids = tree.child_ids(parent)
        if kids and parent not in moved:
            out.append((parent, list(kids)))
    return out


# ---- the sheet ----------------------------------------------------------------------------------


def sheet_questions(
    ctx: Context, settings: tm.SmcSettings, count: int = SHEET_QUESTIONS
) -> list[Candidate]:
    """``count`` questions chosen adaptively against a simulated respondent who answers each one
    A or B with probability 1/2 (``PCG64(seed).jumped(2)``). Each answer is drawn among the
    questions not yet implied, so it never empties the particles. The stop rule is not applied:
    the sheet is fixed in advance and the person answers all of it."""
    state = tm.SmcState.start(ctx.prior, settings)
    pool = candidates(ctx)
    contrasts = np.array([c.contrast for c in pool])
    coin = np.random.Generator(np.random.PCG64(settings.seed).jumped(2))
    asked: set[int] = set()
    out: list[Candidate] = []
    for _ in range(count):
        probs = tm.answer_probabilities(state.weights, contrasts)
        pick = tm.select_question(probs, asked)
        if pick is None:
            break
        asked.add(pick)
        out.append(pool[pick])
        sign = 1.0 if coin.random() < 0.5 else -1.0
        if state.add(sign * contrasts[pick], settings.moves) == 0:  # pragma: no cover
            state.add(-sign * contrasts[pick], settings.moves)
    return out


def sheet_csv(ctx: Context, questions: list[Candidate]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SHEET_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for n, q in enumerate(questions, start=1):
        leaves = list(q.a)
        row = {"id": f"q{n}", "choice": "", "note": ""}
        for k, leaf in enumerate(leaves, start=1):
            row[f"criterion_{k}_id"] = leaf
            row[f"criterion_{k}"] = ctx.names[leaf]
            row[f"a_level_{k}"] = _shown(ctx, leaf, q.a[leaf])
            row[f"b_level_{k}"] = _shown(ctx, leaf, q.b[leaf])
        writer.writerow({k: _safe(str(v)) for k, v in row.items()})
    return buffer.getvalue()


def _safe(value: str) -> str:
    return "'" + value if value.startswith(_FORMULA) else value


def sheet_html(ctx: Context, questions: list[Candidate], settings: tm.SmcSettings) -> str:
    e = html.escape
    info = ctx.prior_info
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Trade-off questions</title>",
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;color:#1b2430}h2{margin-top:2.5rem}table{border-collapse:collapse;"
        "width:100%}td,th{border:1px solid #c8d2de;padding:.35rem .5rem;vertical-align:top;"
        "text-align:left}.rubric{color:#3d4a5a}.better{font-weight:600}</style>",
        "</head><body>",
        f"<h1>Trade-off questions: {e(ctx.project.title)}</h1>",
        f"<p>{len(questions)} questions for the profile <strong>{e(ctx.profile.name)}</strong>. "
        "Each compares two ventures that are the same on every criterion except the two shown. "
        "For each row, write <code>A</code> or <code>B</code> in the sheet's <code>choice</code> "
        "column for the one you would rather back, or <code>skip</code> if you cannot say. Save "
        "it as CSV UTF-8.</p>",
        "<p>Answer from your own judgement, not from the weights you stated: the point is to see "
        "whether the two agree.</p>",
        "<h2>How the questions were chosen</h2>",
        f'<p class="rubric">Adaptively, against a simulated respondent who answers every '
        f"question A or B with probability 1/2. Before each question the prior ({e(info.kind)}, "
        f"{e(info.source)}) is restricted to the simulated answers so far with "
        f"{settings.particles} particles (seed {settings.seed}); the question asked is the one "
        "whose answer is closest to an even split, never one whose answer is already implied "
        "(below 2% or above 98%). The simulated answers are discarded: only the questions are "
        "kept. Your answers then decide the implied weights.</p>",
        "<h2>Questions</h2>",
        "<table><tr><th>#</th><th>Venture A</th><th>Venture B</th><th>Your choice</th></tr>",
    ]
    for n, q in enumerate(questions, start=1):
        cells = []
        for side in (q.a, q.b):
            cells.append(
                "<br>".join(
                    f"{e(ctx.names[k])}: <strong>{e(_shown(ctx, k, side[k]))}</strong>"
                    for k in side
                )
            )
        parts.append(f"<tr><td>q{n}</td><td>{cells[0]}</td><td>{cells[1]}</td><td></td></tr>")
    parts.append("</table>")
    parts.append("<h2>Levels of each criterion asked</h2>")
    used = list(dict.fromkeys(k for q in questions for k in q.a))
    for leaf in used:
        parts.append(f"<h3>{e(ctx.names[leaf])} <code>{e(leaf)}</code></h3>")
        parts.append(
            "<ol>" + "".join(f"<li>{e(lv.shown)}</li>" for lv in ctx.levels[leaf]) + "</ol>"
        )
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


__all__ = [
    "LABEL", "LIMITATIONS", "SHEET_COLUMNS", "SHEET_QUESTIONS", "Candidate", "ConfidenceRow",
    "ConflictOut", "Context", "LeafRow", "Level", "NextOut", "PriorInfo", "QuestionOut",
    "TradeoffFit", "TradeoffReport", "answers_digest", "answers_from_sheet", "build_prior",
    "candidates", "check_answers", "context", "describe", "fit", "next_question",
    "question_key", "sheet_csv", "sheet_html", "sheet_questions", "weight_samples",
]  # fmt: skip
