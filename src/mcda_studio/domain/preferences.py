"""Preference information: swing-weight judgements, scenarios and analysis configuration."""

import math
from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

from pydantic import Field

from mcda_studio.domain.common import DomainModel, FiniteFloat, Id, Issue, Severity
from mcda_studio.domain.criteria import CriterionTree, ValueFunctionSpec
from mcda_studio.domain.pairwise import PairwiseJudgement

#: How sure the person is of their swing ratings. The elicitation check turns it into stated
#: rating intervals by a documented rule (``docs/design/structured-judgement.md``).
Conviction = Literal["high", "medium", "low"]


class GroupElicitation(DomainModel):
    """One sibling group as the person judged it (``parent_id`` None: the top level).

    ``order``: the children best first, each once, no ties; empty when no ranking was given.
    ``pairwise``: optional AHP answers for the same children, a cross-check on the swing ratings.
    ``reasons``: one line per child, in the person's words.
    """

    parent_id: Id | None
    order: list[Id] = Field(default_factory=list, exclude_if=lambda v: not v)
    pairwise: list[PairwiseJudgement] = Field(default_factory=list, exclude_if=lambda v: not v)
    reasons: dict[Id, str] = Field(default_factory=dict, exclude_if=lambda v: not v)


class Elicitation(DomainModel):
    """How a weight profile's judgements were given: who, when, how sure, and why.

    It stores judgements only. The swing ratings stay in ``WeightProfile.judgements``; checks
    (agreement of order and ratings, AHP consistency, rank-order weights) and proposed weight
    uncertainty are derived, never stored.
    """

    elicited_by: str = ""
    elicited_on: date | None = Field(default=None, exclude_if=lambda v: v is None)
    conviction: Conviction | None = Field(default=None, exclude_if=lambda v: v is None)
    note: str = Field(default="", exclude_if=lambda v: not v)
    groups: list[GroupElicitation] = Field(default_factory=list)


class WeightProfile(DomainModel):
    """Raw swing ratings, one per criterion node, relative to that node's siblings.

    A rating expresses the importance of the swing from worst to best *over the scale range
    in use* - not abstract importance. Ratings need not sum to anything; they are normalised
    within each sibling group. An only child needs no rating (its local weight is 1).
    """

    id: Id
    name: str = Field(min_length=1)
    method: str = "swing"
    judgements: dict[Id, FiniteFloat] = Field(default_factory=dict)
    # Schema version 7: how the judgements were given; absent from older files and fingerprints.
    elicitation: Elicitation | None = Field(default=None, exclude_if=lambda v: v is None)


class Scenario(DomainModel):
    """A named variation of the base model: another weight profile and/or other value functions."""

    id: Id
    name: str = Field(min_length=1)
    weight_profile_id: Id | None = None
    value_function_overrides: dict[Id, ValueFunctionSpec] = Field(default_factory=dict)


class AnalysisConfiguration(DomainModel):
    method: str = "mavt"
    weight_profile_id: Id | None = None  # None -> scenario's profile, else the first profile
    scenario_id: Id | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


def check_judgements(
    tree: CriterionTree, judgements: Mapping[str, float], path: str = ""
) -> list[Issue]:
    """Sibling-group rules shared by validation and weighting."""
    issues: list[Issue] = []
    prefix = f"{path}." if path else ""

    def err(code: str, message: str, where: str) -> None:
        issues.append(Issue(severity=Severity.ERROR, code=code, message=message, path=where))

    for key in judgements:
        if key not in tree:
            err(
                "UNKNOWN_JUDGEMENT_NODE",
                f"judgement refers to unknown criterion {key!r}",
                f"{prefix}judgements.{key}",
            )
    parents: list[str | None] = [None, *[i for i in tree.all_ids() if not tree.is_leaf(i)]]
    for pid in parents:
        kids = tree.child_ids(pid)
        if not kids:
            continue
        group = "<top level>" if pid is None else repr(pid)
        ratings: list[float] = []
        group_ok = True
        for kid in kids:
            raw = judgements.get(kid)
            where = f"{prefix}judgements.{kid}"
            if raw is None:
                if len(kids) == 1:
                    ratings.append(1.0)  # only child: weight is necessarily 1
                    continue
                err(
                    "MISSING_JUDGEMENT",
                    f"no swing rating for {kid!r} among siblings of {group}",
                    where,
                )
                group_ok = False
                continue
            if not math.isfinite(raw) or raw < 0:
                err("INVALID_JUDGEMENT", f"swing rating {raw!r} must be finite and >= 0", where)
                group_ok = False
                continue
            ratings.append(float(raw))
        if group_ok and math.fsum(ratings) <= 0.0:
            err(
                "ZERO_SUM_JUDGEMENTS",
                f"all swing ratings under {group} are zero",
                f"{prefix}judgements",
            )
    return issues
