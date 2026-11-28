"""Break-even weights: how far the stated weights must move before the leader changes.

Read-only. Two answers for the current leader of an additive MAVT evaluation:

- **per criterion**: the local weight at which the leader first changes, down and up, with the
  criterion's siblings rescaled in proportion. This is the engine's existing one-way sensitivity
  (``methods/mavt/sensitivity.py::weight_sensitivity``, exact crossovers), reused unchanged and
  only exposed; each threshold is also given as the criterion's share of the whole goal (its local
  weight times its parent's share, which does not move);
- **nearest change**: the global weight vector closest to the stated one in L1 at which another
  alternative ties the leader, and which one (``methods/mavt/break_even.py``). Only leaves with a
  positive stated weight move: a leaf rated 0, such as a screened-only mandate, stays at 0.

For a sorting model it also gives, per alternative, the nearest weight vector that moves it across
its class boundary above and below. Design: ``docs/design/tradeoff-elicitation.md`` (Part 3).
"""

from __future__ import annotations

import numpy as np

from mcda_studio.domain.common import DomainModel
from mcda_studio.domain.errors import McdaError, SensitivityError
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.project import Project
from mcda_studio.methods.mavt.aggregation import ordered_indices, overall_values, prepare_mavt
from mcda_studio.methods.mavt.break_even import nearest_shift
from mcda_studio.methods.mavt.sensitivity import weight_sensitivity
from mcda_studio.methods.mavt.weighting import path_shares
from mcda_studio.methods.sorting.classes import class_index, lower_boundaries

MAVT_METHODS = ("mavt", "mavt_sorting", "smaa2")


class Threshold(DomainModel):
    local_weight: float
    share: float  # of the whole goal
    new_leader: str


class CriterionBreakEven(DomainModel):
    criterion_id: str
    name: str
    local_weight: float
    share: float
    falls_below: Threshold | None  # the leader changes if the weight falls below this
    rises_above: Threshold | None  # ... or rises above this
    sentence: str


class WeightChange(DomainModel):
    criterion_id: str
    name: str
    stated: float
    changed: float


class NearestChange(DomainModel):
    alternative_id: str  # the alternative that ties the leader there
    name: str
    distance: float  # L1 between global weight vectors
    changes: list[WeightChange]  # only the leaves that move


class ClassMove(DomainModel):
    alternative_id: str
    name: str
    class_id: str
    up: NearestChange | None  # into the class above
    down: NearestChange | None  # into the class below


class BreakEven(DomainModel):
    project_fingerprint: str
    weight_profile_id: str
    scenario_id: str | None
    leader: str
    leader_name: str
    criteria: list[CriterionBreakEven]
    nearest: NearestChange | None
    nearest_sentence: str
    challengers: list[NearestChange]
    classes: list[ClassMove]
    rule: str = (
        "Per criterion: the criterion's local weight moves and its siblings are rescaled in "
        "proportion (the engine's one-way sensitivity). Nearest change: the smallest total "
        "movement of global weights (L1), leaves rated 0 held at 0."
    )


def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}%" if x >= 0.0995 else f"{100.0 * x:.1f}%"


def break_even(project: Project, configuration: AnalysisConfiguration | None = None) -> BreakEven:
    cfg = configuration or project.analysis
    if cfg.method not in MAVT_METHODS:
        raise McdaError(
            "break-even weights read the additive value model; choose MAVT, MAVT sorting or "
            "SMAA-2 (whose reference ranking is MAVT's)"
        )
    mavt_cfg = cfg.model_copy(update={"method": "mavt", "parameters": {}})
    prep = prepare_mavt(project, mavt_cfg)
    names = {n.id: n.name for n in project.criteria}
    alt_names = {a.id: a.name for a in project.alternatives}
    alts = prep.alt_ids
    w0 = prep.global_vector()
    values = overall_values(prep.partial, w0)
    order = ordered_indices(values)
    lead = order[0]
    leader = alts[lead]
    shares = path_shares(prep.tree, prep.local)

    criteria: list[CriterionBreakEven] = []
    for node in prep.tree.all_ids():
        parent = prep.tree.node(node).parent_id
        if len(prep.tree.child_ids(parent)) < 2 or shares[node] == 0.0:
            continue
        try:
            result = weight_sensitivity(project, mavt_cfg, node, steps=2)
        except SensitivityError:
            continue
        parent_share = 1.0 if parent is None else shares[parent]

        def threshold(change: object, parent_share: float = parent_share) -> Threshold | None:
            if change is None:
                return None
            t = float(change.weight)  # type: ignore[attr-defined]
            return Threshold(
                local_weight=t,
                share=t * parent_share,
                new_leader=change.to_winner,  # type: ignore[attr-defined]
            )

        down = threshold(result.first_winner_change_down)
        up = threshold(result.first_winner_change_up)
        lead_name = alt_names[leader]
        if down and up:
            sentence = (
                f"{lead_name} stays first unless {names[node]} falls below {_pct(down.share)} "
                f"or rises above {_pct(up.share)}"
            )
        elif down:
            sentence = (
                f"{lead_name} stays first unless {names[node]} falls below {_pct(down.share)}"
            )
        elif up:
            sentence = f"{lead_name} stays first unless {names[node]} rises above {_pct(up.share)}"
        else:
            sentence = f"{lead_name} stays first whatever the weight of {names[node]}"
        criteria.append(
            CriterionBreakEven(
                criterion_id=node,
                name=names[node],
                local_weight=prep.local[node],
                share=shares[node],
                falls_below=down,
                rises_above=up,
                sentence=sentence,
            )
        )

    eligible = w0 > 0.0

    def change(alt: int, d: np.ndarray) -> NearestChange | None:
        shift = nearest_shift(w0, d, eligible)
        if shift is None:
            return None
        moved = [
            WeightChange(
                criterion_id=leaf, name=names[leaf], stated=float(w0[j]),
                changed=float(shift.weights[j]),
            )
            for j, leaf in enumerate(prep.leaf_ids)
            if abs(shift.weights[j] - w0[j]) > 1e-15
        ]  # fmt: skip
        return NearestChange(
            alternative_id=alts[alt], name=alt_names[alts[alt]], distance=shift.distance,
            changes=moved,
        )  # fmt: skip

    challengers = [
        c
        for c in (
            change(i, prep.partial[i] - prep.partial[lead]) for i in range(len(alts)) if i != lead
        )
        if c is not None
    ]
    challengers.sort(key=lambda c: (c.distance, alts.index(c.alternative_id)))
    nearest = challengers[0] if challengers else None

    classes: list[ClassMove] = []
    if project.sorting is not None and cfg.method == "mavt_sorting":
        bounds = lower_boundaries(project.sorting)
        ids = [c.id for c in project.sorting.classes]
        for i, alt in enumerate(alts):
            k = class_index(values[i], bounds)
            raise_to = change(i, prep.partial[i] - bounds[k - 1]) if k > 0 else None
            drop_to = change(i, bounds[k] - prep.partial[i]) if k < len(bounds) else None
            classes.append(
                ClassMove(alternative_id=alt, name=alt_names[alt], class_id=ids[k], up=raise_to,
                          down=drop_to)
            )  # fmt: skip
    if nearest is None:
        nearest_sentence = f"No change of weights makes another alternative tie {alt_names[leader]}"
    else:
        moves = ", ".join(
            f"{c.name} {_pct(c.stated)} to {_pct(c.changed)}" for c in nearest.changes
        )
        nearest_sentence = (
            f"The smallest change that lets another alternative tie {alt_names[leader]} moves "
            f"{_pct(nearest.distance / 2.0)} of the weight ({_pct(nearest.distance)} in L1), and "
            f"{nearest.name} draws level: {moves}"
        )
    return BreakEven(
        project_fingerprint=project.fingerprint(),
        weight_profile_id=prep.profile.id,
        scenario_id=prep.scenario.id if prep.scenario else None,
        leader=leader,
        leader_name=alt_names[leader],
        criteria=criteria,
        nearest=nearest,
        nearest_sentence=nearest_sentence,
        challengers=challengers,
        classes=classes,
    )
